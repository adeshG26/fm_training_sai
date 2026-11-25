"""Typer CLI — a thin wrapper: parse args, call :mod:`pragmatiq.api`, print.

No logic lives here (global rule 1). Command results print to stdout;
progress and log output go to stderr.

Path-like parameters are typed ``str``, not :class:`~pathlib.Path`: any of
them may be a remote URL (``s3://…``), and ``Path("s3://b/k")`` collapses the
double slash to ``s3:/b/k`` — a local path. The api layer accepts both.
"""

from __future__ import annotations

import json
import logging
import sys

import typer

app = typer.Typer(name="pragmatiq", help="pragmatiq: behavioral foundation model for banking events.",
                  no_args_is_help=True, pretty_exceptions_show_locals=False)
synth_app = typer.Typer(help="Synthetic data generation.", no_args_is_help=True)
app.add_typer(synth_app, name="synth")


def _version_callback(value: bool) -> None:
    if value:
        from pragmatiq import __version__

        typer.echo(f"pragmatiq {__version__}")
        raise typer.Exit()


@app.callback()
def _setup(
    ctx: typer.Context,
    verbose: bool = typer.Option(True, "--verbose/--quiet",
                                 help="Show INFO-level progress logs on stderr."),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True,
                                 help="Print the pragmatiq version and exit."),
) -> None:
    ctx.obj = {"verbose": verbose}
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        stream=sys.stderr, format="%(message)s")


@synth_app.command("generate")
def synth_generate(
    out: str = typer.Option("data/synth", help="Output directory."),
    config: str | None = typer.Option(None, help="WorldConfig YAML (configs/data/synthetic.yaml)."),
    n_users: int | None = typer.Option(None, help="Override n_users."),
    seed: int | None = typer.Option(None, help="Override seed."),
    n_workers: int = typer.Option(0, help="Parallel workers (<=1 = inline)."),
    report: bool | None = typer.Option(None, "--report/--no-report",
                                       help="Write realism_report.html (default: only when "
                                            "matplotlib is installed)."),
) -> None:
    """Generate a synthetic banking dataset."""
    from pragmatiq import api

    manifest = api.synthesize(config=config, out=out, n_users=n_users, seed=seed,
                              n_workers=n_workers, write_report=report)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("tokenize")
def tokenize_cmd(
    data_dir: str = typer.Argument(..., help="Generated dataset directory."),
    out: str = typer.Option("data/tokenized", help="Output (shards + tokenizer + index)."),
    config: str | None = typer.Option(None, help="Tokenizer YAML (configs/data/tokenizer.yaml)."),
    tokenizer_dir: str | None = typer.Option(None, help="Reuse an existing tokenizer dir."),
    max_users: int | None = typer.Option(None, help="Cap users (debug)."),
    n_workers: int = typer.Option(0, help="Parallel encode workers (<=1 = inline)."),
) -> None:
    """Fit/apply the tokenizer and write tokenized shards + LMDB index."""
    from pragmatiq import api

    manifest = api.tokenize(data_dir=data_dir, out=out, config=config,
                            tokenizer_dir=tokenizer_dir, max_users=max_users,
                            n_workers=n_workers)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("pretrain")
def pretrain_cmd(
    ctx: typer.Context,
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run_name: str = typer.Option(..., "--name", help="Run name (runs/{name})."),
    model_size: str = typer.Option("small", help="small | medium | large."),
    config: str | None = typer.Option(
        "auto", help="Pretrain YAML (configs/pretrain.yaml), or 'auto' to size the batch and "
                     "schedule from the data and the device."),
    runs_root: str = typer.Option("runs", help="Runs root directory."),
    resume: str | None = typer.Option(None, help="'auto' to resume runs/{name}/checkpoints/last.pt."),
    wandb: bool = typer.Option(False, "--wandb",
                               help="Mirror metrics to Weights & Biases (needs the [tracking] extra)."),
    show_config: bool = typer.Option(False, "--show-config",
                                     help="Print the resolved training + model config as JSON "
                                          "and exit without training."),
) -> None:
    """Pretrain a pragmatiq model (MLM) on tokenized shards."""
    from pragmatiq import api

    # Only forward non-default flags, so `wandb: true` / `verbose:` in the
    # YAML still win over CLI defaults.
    overrides: dict = {}
    if wandb:
        overrides["wandb"] = True
    if ctx.obj and not ctx.obj.get("verbose", True):
        overrides["verbose"] = False  # --quiet also silences the heartbeat
    if show_config:
        plan = api.pretrain_plan(shard_dir, model_size=model_size, config=config, run_name=run_name,
                                 runs_root=runs_root, resume=resume, **overrides)
        typer.echo(json.dumps(plan, indent=2, default=str))
        return
    summary = api.pretrain(shard_dir, run_name, model_size=model_size, config=config,
                           runs_root=runs_root, resume=resume, **overrides)
    typer.echo(json.dumps(summary, indent=2))


@app.command("probe")
def probe_cmd(
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run: str = typer.Option(..., help="Run directory of a trained model."),
    label: str = typer.Option(..., help="Label parquet (labels/<task>.parquet)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    probe_model: str = typer.Option("gbdt", help="Probe head: gbdt | logistic | lightgbm."),
    seed: int = typer.Option(0, help="Probe random seed (for reproducible AUCs)."),
    staleness_window: str | None = typer.Option(
        None, help="Drop the most recent window of history before each eval point "
                   "(e.g. 6h, 1d): the paper's event-staleness robustness check."),
) -> None:
    """Probe a trained model on a label table; reports ROC-AUC + PR-AUC vs baseline."""
    from pragmatiq import api

    typer.echo(json.dumps(
        api.probe(shard_dir, run, label, device=device, probe_model=probe_model, seed=seed,
                  staleness_window=staleness_window), indent=2))


@app.command("uplift")
def uplift_cmd(
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run: str = typer.Option(..., help="Run directory of a trained model."),
    label: str = typer.Option(..., help="comm_uplift label parquet (labels/comm_uplift.parquet)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    learner: str = typer.Option("t", help="uplift meta-learner: t (two-model) | s (single-model)."),
) -> None:
    """Evaluate campaign uplift on a trained model; reports Qini vs an oracle Qini."""
    from pragmatiq import api

    typer.echo(json.dumps(api.uplift(shard_dir, run, label, device=device, learner=learner), indent=2))


@app.command("finetune")
def finetune_cmd(
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run: str = typer.Option(..., help="Run directory of a trained model."),
    label: str = typer.Option(..., help="Label parquet (labels/<task>.parquet)."),
    config: str | None = typer.Option(None, help="Finetune YAML (see configs/finetune/credit.yaml)."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """LoRA fine-tune a trained model's adapters + head on a label table."""
    from pragmatiq import api

    typer.echo(json.dumps(api.finetune(shard_dir, run, label, config=config, device=device), indent=2))


@app.command("embed")
def embed_cmd(
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run: str = typer.Option(..., help="Run directory of a trained model."),
    out: str = typer.Option("embeddings.parquet", help="Output parquet."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """Embed all users to a parquet of user_id + embedding."""
    from pragmatiq import api

    typer.echo(json.dumps(api.embed(shard_dir, run, out=out, device=device), indent=2))


@app.command("quickstart")
def quickstart_cmd(
    out: str = typer.Option("runs/quickstart", help="Output directory."),
    n_users: int = typer.Option(50000, help="Synthetic users."),
    model_size: str = typer.Option("nano", help="Model size for the smoke run."),
    max_steps: int = typer.Option(400, help="Pretrain steps."),
    n_workers: int = typer.Option(0, help="Generation workers."),
) -> None:
    """End-to-end smoke: synth -> tokenize -> pretrain -> probe (prints AUC)."""
    from pragmatiq import api

    res = api.quickstart(out=out, n_users=n_users, model_size=model_size,
                         max_steps=max_steps, n_workers=n_workers)
    typer.echo(json.dumps(res, indent=2))
    typer.echo(res["message"])


@app.command("info")
def info_cmd() -> None:
    """Describe this installation: versions, device, kernels, extras, env vars."""
    from pragmatiq import api

    typer.echo(json.dumps(api.info(), indent=2))


@app.command("validate")
def validate_cmd(
    data_dir: str = typer.Argument(..., help="Raw dataset directory."),
) -> None:
    """Validate a raw dataset against the data contract."""
    from pragmatiq import api

    res = api.validate(data_dir)
    typer.echo(res["summary"])
    raise typer.Exit(code=0 if res["ok"] else 1)


@app.command("export")
def export_cmd(
    run: str = typer.Option(..., help="Run directory of a trained model."),
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory (for an example user)."),
    out: str = typer.Option("pragmatiq_embedder.onnx", help="Output ONNX path."),
    device: str = typer.Option("auto", help="Accepted for symmetry; the graph is built on CPU."),
) -> None:
    """Export the padded-embedder ONNX variant (varlen caveat documented)."""
    from pragmatiq import api

    typer.echo(json.dumps(api.export(run, shard_dir, out=out, device=device), indent=2))


@app.command("benchmark")
def benchmark_cmd(
    run: str = typer.Option(..., help="Run directory of a trained model."),
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
    out: str = typer.Option("benchmark_results.md", help="Results markdown."),
    precision: str = typer.Option("auto", help="auto | bf16 | fp32 (CUDA autocast dtype)."),
) -> None:
    """Benchmark batch-embedding throughput; writes RESULTS.md."""
    from pragmatiq import api

    typer.echo(json.dumps(api.benchmark(run, shard_dir, device=device, out=out, precision=precision),
                          indent=2))


@app.command("gnn")
def gnn_cmd(
    shard_dir: str = typer.Argument(..., help="Tokenized shard directory."),
    run: str = typer.Option(..., help="Run directory of a trained model."),
    transfers: str = typer.Option(..., help="transfers.parquet."),
    aml_label: str = typer.Option(..., help="labels/aml.parquet."),
    seeds: str = typer.Option("0,1,2", help="Comma-separated seeds."),
    epochs: int = typer.Option(150, help="GraphSAGE epochs."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """Run the four-arm AML GNN ablation (isolated vs GNN+PRAGMA vs GNN+handcrafted vs no-graph control)."""
    from pragmatiq import api

    seed_tuple = tuple(int(s) for s in seeds.split(","))
    res = api.gnn(shard_dir, run, transfers, aml_label, seeds=seed_tuple, epochs=epochs, device=device)
    typer.echo(json.dumps(res, indent=2))


runs_app = typer.Typer(help="Inspect runs.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(runs_root: str = typer.Option("runs", help="Runs root.")) -> None:
    """List runs with their last logged step/loss."""
    from pragmatiq import api

    typer.echo(json.dumps(api.runs_list(runs_root), indent=2))


@runs_app.command("compare")
def runs_compare(
    names: list[str] = typer.Argument(..., help="Run names to compare."),
    runs_root: str = typer.Option("runs", help="Runs root."),
) -> None:
    """Compare the last metrics of several runs side by side."""
    from pragmatiq import api

    typer.echo(json.dumps(api.runs_compare(names, runs_root), indent=2))


@synth_app.command("calibrate")
def synth_calibrate(
    stats: str = typer.Option(..., help="aggregates.yaml with bank-shareable statistics."),
    config: str | None = typer.Option(None, help="Base WorldConfig YAML to start from."),
    out: str | None = typer.Option(None, help="Where to write the calibrated config YAML."),
) -> None:
    """Fit generator priors to aggregate statistics (moment matching)."""
    from pragmatiq import api

    result = api.calibrate(stats=stats, config=config, out=out)
    typer.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------- real data (intake)

intake_app = typer.Typer(help="Real client tables -> the data contract (replaces `synth`).",
                         no_args_is_help=True)
app.add_typer(intake_app, name="intake")

_SOURCE_HELP = "Client tables: folder, s3:// URL or connector name (default: mapping's source:)."
_NOW_HELP = "Pin the run clock, e.g. 2028-10-01 (default: mapping's options.now, else now)."


@intake_app.command("doctor")
def intake_doctor_cmd(
    device: str = typer.Option("auto", help="auto | cuda | cpu (auto honours PRAGMATIQ_DEVICE)."),
    work_dir: str = typer.Option(".", help="Where staging would go, to report free disk."),
) -> None:
    """Show the machine and prove the GPU path works."""
    from pragmatiq import api

    typer.echo(json.dumps(api.intake_doctor(device=device, work_dir=work_dir), indent=2))


@intake_app.command("discover")
def intake_discover_cmd(
    source: str = typer.Argument(..., help="Folder / URL / connector name of the client tables."),
    out: str | None = typer.Option(None, help="Where to write the DRAFT mapping YAML."),
    client: str = typer.Option("", help="Client name for the mapping."),
    connectors: str | None = typer.Option(None, help="connectors.yaml with named sources."),
) -> None:
    """Scan client tables and draft a mapping (review it before running)."""
    from pragmatiq import api

    res = api.intake_discover(source, out=out, client=client, connectors=connectors)
    typer.echo(json.dumps(res, indent=2))


@intake_app.command("check")
def intake_check_cmd(
    mapping: str = typer.Option(..., help="Intake mapping YAML."),
    source: str | None = typer.Option(None, help=_SOURCE_HELP),
    out: str | None = typer.Option(None, help="Where to write the quality report."),
    connectors: str | None = typer.Option(None, help="connectors.yaml with named sources."),
    device: str = typer.Option("auto", help="auto | cuda | cpu."),
    now: str | None = typer.Option(None, help=_NOW_HELP),
    deep_links: bool = typer.Option(True, "--deep-links/--no-deep-links",
                                    help="Resolve child rows to their parents."),
) -> None:
    """Gate the mapping against the data and score readiness /100."""
    from pragmatiq import api

    res = api.intake_check(mapping, source=source, out=out, connectors=connectors, device=device,
                           now=now, deep_links=deep_links)
    typer.echo(json.dumps(res, indent=2))
    raise typer.Exit(code=0 if res["ok"] else 1)


@intake_app.command("run")
def intake_run_cmd(
    mapping: str = typer.Option(..., help="Intake mapping YAML."),
    out: str = typer.Option("data/client", help="Output dir (events/profiles parquet)."),
    source: str | None = typer.Option(None, help=_SOURCE_HELP),
    connectors: str | None = typer.Option(None, help="connectors.yaml with named sources."),
    device: str = typer.Option("auto", help="auto | cuda | cpu."),
    now: str | None = typer.Option(None, help=_NOW_HELP),
    skip_check: bool = typer.Option(False, help="Skip the mapping gate."),
    work_dir: str | None = typer.Option(None, help="Staging dir (default <out>/_work)."),
    resume: bool = typer.Option(False, help="Continue a stopped run."),
    keep_work: bool = typer.Option(False, help="Keep staged partitions afterwards."),
) -> None:
    """Transform client tables into events.parquet + profiles.parquet."""
    from pragmatiq import api

    res = api.intake_run(mapping, out, source=source, connectors=connectors, device=device,
                         now=now, skip_check=skip_check, work_dir=work_dir, resume=resume,
                         keep_work=keep_work)
    typer.echo(json.dumps(res, indent=2))


@intake_app.command("report")
def intake_report_cmd(
    out: str = typer.Argument(..., help="The directory `intake run` wrote to."),
    mapping: str | None = typer.Option(None, help="Mapping, so numeric fields report as numeric."),
    rows_in: int = typer.Option(0, help="Source row count for the reconciliation line."),
) -> None:
    """Reconciliation + tokenizer vocabulary forecast for a finished run."""
    from pragmatiq import api

    typer.echo(json.dumps(api.intake_report(out, mapping=mapping, rows_in=rows_in), indent=2))


@intake_app.command("labels")
def intake_labels_cmd(
    data_dir: str = typer.Argument(..., help="Intake output dir (holds events.parquet)."),
    labels: str = typer.Option(..., help="Labels YAML (eval_ts + tasks)."),
    mapping: str | None = typer.Option(None, help="Mapping (for source: and timezone)."),
    source: str | None = typer.Option(None, help=_SOURCE_HELP),
) -> None:
    """Build forecast labels/<task>.parquet from the client tables."""
    from pragmatiq import api

    res = api.intake_labels(data_dir, labels, source=source, mapping=mapping)
    typer.echo(json.dumps(res, indent=2))


@app.command("prepare")
def prepare_cmd(
    mapping: str = typer.Option(..., help="Intake mapping YAML."),
    labels: str | None = typer.Option(None, help="Labels YAML (omit to skip labels)."),
    out: str = typer.Option("data/client", help="Output dataset dir."),
    source: str | None = typer.Option(None, help=_SOURCE_HELP),
    connectors: str | None = typer.Option(None, help="connectors.yaml with named sources."),
    device: str = typer.Option("auto", help="auto | cuda | cpu."),
    now: str | None = typer.Option(None, help=_NOW_HELP),
    allow_red: bool = typer.Option(False, help="Proceed even when readiness is RED."),
) -> None:
    """Real data -> validated dataset: intake check + run + report + labels + validate."""
    from pragmatiq import api

    res = api.prepare(mapping, out=out, labels=labels, source=source, connectors=connectors,
                      device=device, now=now, allow_red=allow_red)
    typer.echo(json.dumps(res, indent=2))


@app.command("pipeline")
def pipeline_cmd(
    mapping: str = typer.Option(..., help="Intake mapping YAML."),
    labels: str | None = typer.Option(None, help="Labels YAML."),
    out: str = typer.Option("runs/client", help="Root for raw/ tok/ runs/ embeddings."),
    source: str | None = typer.Option(None, help=_SOURCE_HELP),
    run_name: str = typer.Option("client", help="Pretrain run name."),
    model_size: str = typer.Option("nano", help="nano (CPU smoke) | small | base ..."),
    config: str | None = typer.Option(None, help="Pretrain config YAML, or 'auto'."),
    max_steps: int | None = typer.Option(None, help="Override pretrain max_steps."),
    tokenizer_config: str | None = typer.Option(None, help="Tokenizer YAML."),
    n_workers: int = typer.Option(0, help="Parallel tokenize workers."),
    device: str = typer.Option("auto", help="auto | cuda | cpu."),
    skip_prepare: bool = typer.Option(False, help="Reuse an existing <out>/raw."),
) -> None:
    """Real data end to end: prepare -> tokenize -> pretrain -> embed -> probe."""
    from pragmatiq import api

    pcfg: str | dict | None = config
    if max_steps is not None:
        from pragmatiq.core.config import load_yaml

        base = load_yaml(config) if config and config != "auto" else {}
        pcfg = {**base, "max_steps": max_steps, "warmup_steps": max(10, max_steps // 10)}
    res = api.pipeline(mapping, labels=labels, out=out, source=source, run_name=run_name,
                       model_size=model_size, pretrain_config=pcfg,
                       tokenizer_config=tokenizer_config, n_workers=n_workers, device=device,
                       skip_prepare=skip_prepare)
    typer.echo(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    app()
