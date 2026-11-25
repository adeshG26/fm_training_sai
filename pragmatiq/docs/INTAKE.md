# Real data with `pragmatiq intake`

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

`pragmatiq intake` turns a client's own tables into the same parquet contract
the synthetic generator writes, so everything downstream (`tokenize`,
`pretrain`, `embed`, `probe`, `finetune`) runs unchanged.

```
client tables ──► intake check ──► intake run ──► intake labels ──► validate ──► tokenize ──► pretrain ──► embed / probe
 (folder / s3)    gate + score     events.parquet  labels/*.parquet
                  /100             profiles.parquet
```

`pragmatiq prepare` runs the first four steps in one go; `pragmatiq pipeline`
runs all of them.

## Install

```bash
pip install -e ".[intake,train]"     # + [s3] to read/write s3://, [intake-gpu] on a CUDA Linux box
pragmatiq intake doctor              # RAM / disk / GPU intake will use
```

## 1. Map the client's tables (once per client)

```bash
pragmatiq intake discover ../data --client acme --out configs/intake/acme.draft.yaml
```

`discover` guesses each table's shape and drops identifier columns. **Review the
draft** before using it — it is a guess. Things to check:

| Check | Why |
| --- | --- |
| `source_name` is one of `transaction`, `app`, `trading`, `communication` | Anything else is rejected (and would otherwise be tokenized as `transaction`). |
| Profile (`entity_master`) fields hold nothing that changes after a label's cut | Events are truncated at `eval_ts`; **profiles are not**. Drop closure dates, "is active" flags, latest risk ratings. |
| Date columns used as *values* are dropped | Each is coarsened to a day and adds ~1,000 strings to the vocabulary. |
| Amounts / scores are listed under `numeric` | Reported as percentile-binned magnitudes. |
| `options.now` is set when the data is dated past today | Otherwise later rows are dropped as "future events". |

`configs/intake/client.yaml` is a reviewed example for the L&D client tables.

## 2. Declare the labels

The synthetic world simulates outcomes; real data records them as rows. A label
is a SQL predicate on a client table, observed in a window after a cut:

```yaml
eval_ts: "2027-10-01"          # history before, outcome window after
tasks:
  default_12m:
    table: product_snapshot
    time: snapshot_date
    where: "days_delinquent >= 90"
    horizon_days: 365
    exclude_prior: true        # already bad at the cut -> not scored
```

Each task becomes `labels/<task>.parquet` with `user_id, eval_ts, label`. See
`configs/intake/labels.yaml`.

## 3. Run

```bash
# data only: check -> run -> report -> labels -> validate
pragmatiq prepare --mapping configs/intake/client.yaml \
                  --labels  configs/intake/labels.yaml --out data/client

# then the usual pipeline
pragmatiq tokenize data/client --out data/client_tok --n-workers 8
pragmatiq pretrain data/client_tok --name client --model-size small --config auto
pragmatiq embed    data/client_tok --run runs/client --out embeddings.parquet
pragmatiq probe    data/client_tok --run runs/client --label data/client/labels/default_12m.parquet
```

Or everything at once (a CPU-sized smoke run):

```bash
pragmatiq pipeline --mapping configs/intake/client.yaml --labels configs/intake/labels.yaml \
                   --out runs/client --model-size nano --max-steps 400
```

Individual steps are available as `pragmatiq intake check | run | report | labels`
(`--resume` continues a stopped `run`). All of it is callable from Python as
`pragmatiq.api.prepare(...)`, `api.pipeline(...)`, `api.intake_run(...)`, etc.

## Remote data

Any source or output may be an `s3://` URL (install `[s3]`). Named sources with
required credentials can be declared in a `connectors.yaml` and passed with
`--connectors`.

## Output

| Path | What it is |
| --- | --- |
| `events.parquet` | One row per customer event: `user_id, ts, source, fields` |
| `profiles.parquet` | One row per customer: attributes + lifelong milestones |
| `labels/<task>.parquet` | Forecast labels at `eval_ts` |
| `quality_report.json` | Readiness score /100 and what cost points |
| `output_report.json` | Row reconciliation (in / events / dropped by reason) and vocabulary forecast |

`transfers.parquet` (the AML graph) is not produced: it needs
customer-to-customer transfers, which the client tables do not carry.
