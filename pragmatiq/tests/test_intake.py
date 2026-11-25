"""Real-data intake: client tables -> the parquet contract + forecast labels.

Builds a tiny client (profile master, transactions + a merchant dimension, a
product snapshot and a fraud case table) and runs the public ``prepare`` path,
asserting the output is exactly the contract ``synthesize`` writes.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("polars")
yaml = pytest.importorskip("yaml")

from pragmatiq import api  # noqa: E402
from pragmatiq.core.errors import ConfigError, DataContractError  # noqa: E402
from pragmatiq.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA, SOURCES, label_schema  # noqa: E402

EVAL = datetime(2027, 1, 1)


def _write(root: Path, name: str, table: dict) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), root / name / "part-0.parquet")


@pytest.fixture()
def client(tmp_path: Path) -> dict[str, Path]:
    """Six customers. c2 goes 90+ DPD after the cut, c3 was already 90+ before it,
    c4 gets a fraud case after the cut, c6 only has activity after the cut."""
    src = tmp_path / "tables"
    _write(src, "customer_profile", {
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "segment": ["retail", "retail", "smb", "retail", "smb", "retail"],
        "open_date": [date(2025, 1, d) for d in range(1, 7)],
        "is_active": [True, True, False, True, True, True],
    })
    ts = [datetime(2026, 3, 1, 10), datetime(2026, 6, 2, 11), datetime(2026, 9, 3, 12),
          datetime(2026, 11, 4, 13), datetime(2026, 12, 5, 14), datetime(2027, 3, 6, 15),
          datetime(2028, 6, 1, 9),  # kept under options.now=2028-12-31, dropped at now=2027-06-01
          datetime(2026, 7, 7, 7)]
    _write(src, "txn_event", {
        "txn_id": [f"t{i}" for i in range(8)],
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6", "c1", ""],  # "" -> dropped
        "event_ts": pa.array(ts, type=pa.timestamp("us")),
        "amount": [10.5, 22.0, 3.25, 99.0, 5.0, 7.5, 12.0, 1.0],
        "merchant_id": ["m1", "m2", "m1", "m2", "m1", "m2", "m1", "m1"],
    })
    _write(src, "merchant", {"merchant_id": ["m1", "m2"], "primary_mcc": ["5411", "5812"]})
    _write(src, "product_snapshot", {
        "customer_id": ["c1", "c2", "c2", "c3", "c3", "c4", "c5"],
        "product_id": ["p1", "p2", "p2", "p3", "p3", "p4", "p5"],
        "snapshot_date": [date(2026, 6, 30), date(2026, 6, 30), date(2027, 6, 30),
                          date(2026, 6, 30), date(2026, 12, 31), date(2026, 6, 30),
                          date(2026, 6, 30)],
        "days_delinquent": [0, 0, 95, 30, 120, 0, 0],
    })
    _write(src, "fraud_case", {
        "alert_id": ["a1", "a2"],
        "customer_id": ["c4", "c5"],
        "alert_ts": pa.array([datetime(2027, 4, 1), datetime(2026, 5, 1)], type=pa.timestamp("us")),
        "fraud_case_id": ["f1", None],
    })

    mapping = {
        "client": "tiny", "source": str(src),
        "options": {"timezone": "UTC", "drop_future_events": True, "now": "2028-12-31"},
        "tables": {
            "customer_profile": {"shape": "entity_master", "entity": "customer_id",
                                 "lifelong": ["open_date"], "fields": {"drop": ["is_active"]}},
            "txn_event": {"shape": "event_stream", "entity": "customer_id", "time": "event_ts",
                          "source_name": "transaction",
                          "fields": {"drop": ["txn_id", "merchant_id"], "numeric": ["amount"]}},
            "merchant": {"shape": "dimension", "action": "enrich", "enrich_into": "txn_event",
                         "enrich_on": "merchant_id", "fields": {"include": ["primary_mcc"]}},
            "product_snapshot": {"shape": "periodic_snapshot", "entity": "customer_id",
                                 "time": "snapshot_date", "source_name": "transaction",
                                 "snapshot_of": "product_id", "fields": {"drop": ["product_id"]}},
            "fraud_case": {"shape": "event_stream", "entity": "customer_id", "time": "alert_ts",
                           "source_name": "communication",
                           "fields": {"drop": ["alert_id", "fraud_case_id"]}},
        },
    }
    labels = {
        "eval_ts": EVAL.isoformat(),
        "tasks": {
            "default_12m": {"table": "product_snapshot", "time": "snapshot_date",
                            "where": "days_delinquent >= 90", "horizon_days": 365},
            "fraud_12m": {"table": "fraud_case", "time": "alert_ts",
                          "where": "fraud_case_id IS NOT NULL", "horizon_days": 365,
                          "exclude_prior": False},
        },
    }
    mp, lp = tmp_path / "mapping.yaml", tmp_path / "labels.yaml"
    mp.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    lp.write_text(yaml.safe_dump(labels), encoding="utf-8")
    return {"src": src, "mapping": mp, "labels": lp, "root": tmp_path, "raw_mapping": mapping}


def _labels(path: Path) -> dict[str, int]:
    t = pq.read_table(path).to_pydict()
    return dict(zip(t["user_id"], t["label"]))


def test_prepare_writes_the_contract(client: dict) -> None:
    out = client["root"] / "out"
    res = api.prepare(client["mapping"], out=out, labels=client["labels"], device="cpu")

    assert res["validate"]["ok"]
    assert pq.read_schema(out / "events.parquet").equals(EVENTS_SCHEMA)
    assert pq.read_schema(out / "profiles.parquet").equals(PROFILES_SCHEMA)

    events = pq.read_table(out / "events.parquet").to_pylist()
    assert {e["source"] for e in events} <= set(SOURCES)
    assert "" not in {e["user_id"] for e in events}           # empty customer id dropped
    txn_fields = [dict(e["fields"]) for e in events if "amount" in dict(e["fields"])]
    assert txn_fields and all(f["primary_mcc"] in {"5411", "5812"} for f in txn_fields)

    profiles = {p["user_id"]: p for p in pq.read_table(out / "profiles.parquet").to_pylist()}
    assert set(profiles) == {"c1", "c2", "c3", "c4", "c5", "c6"}
    assert "is_active" not in dict(profiles["c1"]["attributes"])  # dropped by the mapping
    assert [m["key"] for m in profiles["c1"]["lifelong"]] == ["open_date"]


def test_labels_are_point_in_time(client: dict) -> None:
    out = client["root"] / "out"
    res = api.prepare(client["mapping"], out=out, labels=client["labels"], device="cpu")

    for task in ("default_12m", "fraud_12m"):
        schema = pq.read_schema(out / "labels" / f"{task}.parquet")
        assert schema.equals(label_schema("default_12m")), task   # forecast contract
        ts = pq.read_table(out / "labels" / f"{task}.parquet")["eval_ts"].to_pylist()
        assert set(ts) == {EVAL}

    default = _labels(out / "labels" / "default_12m.parquet")
    # c6 has no history before the cut; c3 was already 90+ DPD at the cut
    assert "c6" not in default and "c3" not in default
    assert default == {"c1": 0, "c2": 1, "c4": 0, "c5": 0}
    assert res["labels"]["tasks"]["default_12m"]["excluded_prior"] == 1

    fraud = _labels(out / "labels" / "fraud_12m.parquet")
    assert fraud["c4"] == 1 and fraud["c5"] == 0  # c5's alert never became a case


def test_prepare_is_deterministic(client: dict) -> None:
    def digest(d: Path) -> dict[str, str]:
        return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.rglob("*.parquet"))}

    a, b = client["root"] / "a", client["root"] / "b"
    api.prepare(client["mapping"], out=a, labels=client["labels"], device="cpu")
    api.prepare(client["mapping"], out=b, labels=client["labels"], device="cpu")
    assert digest(a) == digest(b)


def test_pinned_clock_keeps_future_dated_rows(client: dict) -> None:
    kept = api.intake_run(client["mapping"], client["root"] / "kept", device="cpu")
    early = api.intake_run(client["mapping"], client["root"] / "early", device="cpu",
                           now="2027-06-01")
    assert early["events_written"] < kept["events_written"]


def test_unknown_source_name_is_rejected(client: dict) -> None:
    m = client["raw_mapping"]
    m["tables"]["txn_event"]["source_name"] = "card_payments"
    bad = client["root"] / "bad.yaml"
    bad.write_text(yaml.safe_dump(m), encoding="utf-8")
    with pytest.raises(ConfigError, match="card_payments"):
        api.intake_check(bad, device="cpu")


def test_mapping_that_does_not_match_data_fails_prepare(client: dict) -> None:
    m = client["raw_mapping"]
    m["tables"]["txn_event"]["time"] = "no_such_column"
    bad = client["root"] / "bad.yaml"
    bad.write_text(yaml.safe_dump(m), encoding="utf-8")
    with pytest.raises(DataContractError, match="does not match"):
        api.prepare(bad, out=client["root"] / "out", device="cpu")


@pytest.mark.parametrize("cfg, msg", [
    ({"tasks": {"x": {}}}, "eval_ts"),
    ({"eval_ts": "2027-01-01", "tasks": {"fraud": {"table": "t", "time": "t", "where": "1=1",
                                                   "horizon_days": 30}}}, "non-forecast"),
    ({"eval_ts": "2027-01-01", "tasks": {"d": {"table": "t"}}}, "needs 'time'"),
])
def test_labels_config_errors(cfg: dict, msg: str) -> None:
    from pragmatiq.data.intake.labels import load_labels_config

    with pytest.raises(ConfigError, match=msg):
        load_labels_config(cfg)
