"""Forecast label tables from client tables: the part of the synthetic generator
that intake has no equivalent for.

The synthetic world writes ``labels/*.parquet`` itself. Real data carries
outcomes as ordinary rows (a delinquency in a product snapshot, a fraud case
being opened), so a label is declared, not simulated:

    eval_ts: "2027-10-01"            # the point-in-time cut
    population: {min_events: 1}      # who is scored: users with history before the cut
    tasks:
      default_12m:
        table: product_snapshot      # client table holding the outcome
        entity: customer_id
        time: snapshot_date
        where: "days_delinquent >= 90"   # SQL predicate (polars SQL) marking a bad row
        horizon_days: 365            # outcome window (eval_ts, eval_ts + horizon]
        exclude_prior: true          # drop users already bad at the cut

Every task is a forecast table in pragmatiq's contract
(``user_id``, ``eval_ts``, ``label``): histories are truncated at ``eval_ts``
before embedding, so the outcome window never leaks into a user's embedding.
User ids and times go through the same expressions intake's transform uses,
so a label's ``user_id`` always matches the ``user_id`` in ``events.parquet``.
Output is sorted by ``user_id`` and therefore byte-identical across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pragmatiq.core.errors import ConfigError, DataContractError
from pragmatiq.core.schema import LABEL_TASKS, label_schema

from .shapes import entity_expr, time_expr
from .tables import scan_table

# label tasks whose contract is not "one row per user at an eval point": a
# forecast built here must not be written under one of these names
_NON_FORECAST = {t for t in LABEL_TASKS if "eval_ts" not in label_schema(t).names}


@dataclass
class LabelTask:
    """One forecast label: which rows count as the outcome, and in what window."""

    name: str
    table: str
    time: str
    where: str
    horizon_days: int
    entity: str = "customer_id"
    exclude_prior: bool = True


@dataclass
class LabelsConfig:
    """The cut, the scored population and the tasks to build."""

    eval_ts: datetime
    tasks: list[LabelTask] = field(default_factory=list)
    min_events: int = 1
    timezone: str = "UTC"


def _parse_ts(value: Any, what: str) -> datetime:
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{what}: {value!r} is not an ISO date/time") from exc
    if ts.tzinfo is not None:
        raise ConfigError(f"{what}: give a naive UTC time, not {value!r}")
    return ts


def load_labels_config(path: str | Path | dict[str, Any], timezone: str = "UTC") -> LabelsConfig:
    """Read and check a labels YAML (or an equivalent dict)."""
    if isinstance(path, dict):
        raw, where = path, "labels config"
    else:
        import yaml

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"labels config not found: {p}")
        raw, where = yaml.safe_load(p.read_text(encoding="utf-8")) or {}, str(p)

    if "eval_ts" not in raw:
        raise ConfigError(f"{where}: 'eval_ts' is required")
    tasks_raw = raw.get("tasks") or {}
    if not tasks_raw:
        raise ConfigError(f"{where}: no tasks declared")

    tasks = []
    for name, spec in tasks_raw.items():
        spec = spec or {}
        if name in _NON_FORECAST:
            raise ConfigError(
                f"{where}: task {name!r} has a non-forecast contract in pragmatiq "
                f"({label_schema(name).names}); name this forecast differently, e.g. "
                f"{name}_12m")
        for need in ("table", "time", "where", "horizon_days"):
            if need not in spec:
                raise ConfigError(f"{where}: task {name!r} needs '{need}'")
        horizon = int(spec["horizon_days"])
        if horizon <= 0:
            raise ConfigError(f"{where}: task {name!r} horizon_days must be positive")
        tasks.append(LabelTask(name=name, table=str(spec["table"]), time=str(spec["time"]),
                               where=str(spec["where"]), horizon_days=horizon,
                               entity=str(spec.get("entity", "customer_id")),
                               exclude_prior=bool(spec.get("exclude_prior", True))))

    population = raw.get("population") or {}
    return LabelsConfig(eval_ts=_parse_ts(raw["eval_ts"], f"{where}: eval_ts"),
                        tasks=tasks,
                        min_events=int(population.get("min_events", 1)),
                        timezone=str(raw.get("timezone", timezone)))


def _population(events_path: Path, eval_ts: datetime, min_events: int) -> pl.DataFrame:
    """Users with at least ``min_events`` events strictly before the cut."""
    if not events_path.exists():
        raise DataContractError(f"{events_path} not found; run `pragmatiq intake run` first")
    return (pl.scan_parquet(events_path)
              .select("user_id", "ts")
              .filter(pl.col("ts") < eval_ts)
              .group_by("user_id").len()
              .filter(pl.col("len") >= min_events)
              .select("user_id")
              .collect())


def _outcome_rows(source_dir: Path, task: LabelTask, timezone: str) -> pl.LazyFrame:
    """(user_id, t) for every row of the task's table matching its predicate."""
    try:
        lf = scan_table(source_dir, task.table)
    except Exception as exc:  # noqa: BLE001 - re-raised with the task named
        raise DataContractError(f"label task {task.name!r}: cannot read table "
                                f"{task.table!r} under {source_dir}: {exc}") from exc
    schema = lf.collect_schema()
    for col in (task.entity, task.time):
        if col not in schema:
            raise ConfigError(f"label task {task.name!r}: column {col!r} not in "
                              f"{task.table!r} (has {sorted(schema)})")
    try:
        pred = pl.sql_expr(task.where)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"label task {task.name!r}: bad 'where' {task.where!r}: {exc}") from exc
    return (lf.filter(pred)
              .select(entity_expr(task.entity, "user_id"),
                      time_expr(task.time, schema[task.time], timezone, "t"))
              .drop_nulls())


def build_labels(source_dir: str | Path, data_dir: str | Path,
                 config: str | Path | dict[str, Any] | LabelsConfig,
                 timezone: str = "UTC") -> dict[str, Any]:
    """Write ``<data_dir>/labels/<task>.parquet`` for every configured task.

    Args:
        source_dir: the client tables (the folder intake read).
        data_dir: the intake output holding ``events.parquet``; labels are
            written to its ``labels/`` folder.
        config: labels YAML path, dict, or a parsed :class:`LabelsConfig`.
        timezone: how naive client timestamps are read (the mapping's).

    Returns:
        ``{"eval_ts", "population", "tasks": {name: {path, rows, positives,
        rate, excluded_prior}}}``.
    """
    cfg = config if isinstance(config, LabelsConfig) else load_labels_config(config, timezone)
    source_dir, data_dir = Path(source_dir), Path(data_dir)
    users = _population(data_dir / "events.parquet", cfg.eval_ts, cfg.min_events)
    out_dir = data_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"eval_ts": cfg.eval_ts.isoformat(), "population": users.height,
                              "tasks": {}}
    for task in cfg.tasks:
        end = cfg.eval_ts + timedelta(days=task.horizon_days)
        rows = _outcome_rows(source_dir, task, cfg.timezone)
        bad_after = (rows.filter((pl.col("t") > cfg.eval_ts) & (pl.col("t") <= end))
                         .select("user_id").unique().collect())
        prior = (rows.filter(pl.col("t") <= cfg.eval_ts).select("user_id").unique().collect()
                 if task.exclude_prior else pl.DataFrame({"user_id": []}, schema={"user_id": pl.String}))

        scored = users.join(prior, on="user_id", how="anti")
        table = (scored.with_columns(pl.col("user_id").is_in(bad_after["user_id"].implode())
                                     .cast(pl.Int8).alias("label"))
                       .with_columns(pl.lit(cfg.eval_ts, dtype=pl.Datetime("us")).alias("eval_ts"))
                       .select("user_id", "eval_ts", "label")
                       .sort("user_id"))

        schema = (label_schema(task.name) if task.name in LABEL_TASKS
                  else label_schema("default_12m"))
        arrow = table.to_arrow().select(schema.names).cast(schema)
        path = out_dir / f"{task.name}.parquet"
        pq.write_table(pa.Table.from_batches(arrow.to_batches(), schema=schema), path,
                       compression="zstd")

        positives = int(table["label"].sum())
        result["tasks"][task.name] = {
            "path": str(path), "rows": table.height, "positives": positives,
            "rate": round(positives / table.height, 4) if table.height else 0.0,
            "excluded_prior": users.height - scored.height,
        }
    return result
