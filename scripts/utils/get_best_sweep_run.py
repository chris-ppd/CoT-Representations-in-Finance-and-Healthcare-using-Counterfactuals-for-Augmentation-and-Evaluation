"""
Print the top-N runs from a WandB sweep, ranked by a chosen metric.

Parameter resolution order (mirrors finetune_bert.py):
  1. wandb run.config  — swept parameters injected by the sweep agent
  2. command field     — fixed CLI args baked into the sweep YAML
  3. 'n/a'             — not recorded anywhere retrievable

Usage
-----
python utils/get_best_sweep_run.py \\
    --sweep-id <entity>/<project>/<sweep_id> \\
    --sweep-config configs/sweeps/finetune_bert_sweep.yaml \\
    [--top-n 3] \\
    [--metric eval/f1_macro]

The sweep_id string is printed by `wandb sweep ...` and also appears in the
WandB dashboard URL:  https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
"""

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch WandB sweep results and print the best run(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        metavar="entity/project/sweep_id",
        help="Full sweep path shown after `wandb sweep` or in the dashboard URL.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the sweep YAML. Columns are built from `parameters` (swept) "
            "and `command` (fixed CLI) fields. Swept params are read from "
            "wandb run.config; fixed CLI params fall back to the YAML values."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top runs to display.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/f1_macro",
        help="WandB summary key to rank runs by (higher is better).",
    )
    return parser.parse_args()


def _parse_sweep_yaml(path: str) -> tuple[list[str], dict[str, str]]:
    """Parse a sweep YAML and return swept param names and fixed CLI values.

    Returns:
        swept_params: Parameter names from the `parameters` field, in order.
        fixed_cli:    {canonical_key: value} for --key value pairs found in
                      `command` that are not already swept parameters. Keys are
                      normalised (--learning-rate → learning_rate).
    """
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        sweep_cfg = yaml.safe_load(f)

    swept_params: list[str] = list(sweep_cfg.get("parameters", {}).keys())
    swept_set = set(swept_params)

    # Walk command tokens and collect --key value pairs
    command: list = sweep_cfg.get("command", [])
    fixed_cli: dict[str, str] = {}
    i = 0
    while i < len(command):
        token = str(command[i])
        if token.startswith("--") and i + 1 < len(command):
            next_token = str(command[i + 1])
            # Skip special tokens (${env}, ${args}) and nested flags
            if not next_token.startswith("${") and not next_token.startswith("--"):
                key = token.lstrip("-").replace("-", "_")
                if key not in swept_set:
                    fixed_cli[key] = next_token
                i += 2
                continue
        i += 1

    return swept_params, fixed_cli


def _resolve_param(run_config: dict, key: str, fixed_cli: dict[str, str]) -> str:
    """Resolve a parameter value using: wandb run.config → fixed CLI → 'n/a'."""
    if key in run_config:
        return str(run_config[key])
    if key in fixed_cli:
        return fixed_cli[key]
    return "n/a"


# Helper function that gives us the best f1 macro after training
def get_best_metric(run, metric="eval/f1_macro"):
    history = run.history(keys=[metric])
    if history.empty or metric not in history.columns:
        return None
    return history[metric].max()


def main() -> None:
    args = _parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()

    try:
        sweep = api.sweep(args.sweep_id)
    except Exception as exc:
        print(f"Could not fetch sweep '{args.sweep_id}': {exc}", file=sys.stderr)
        sys.exit(1)

    # runs = [r for r in sweep.runs if r.summary.get(args.metric) is not None]

    # if not runs:
    #     print(
    #         f"No finished runs with metric '{args.metric}' found in sweep "
    #         f"'{args.sweep_id}'.",
    #         file=sys.stderr,
    #     )
    #     sys.exit(1)

    # runs.sort(key=lambda r: r.summary[args.metric], reverse=True)

    # First compute best metrics for all runs
    runs_with_scores = []
    for r in sweep.runs:
        score = get_best_metric(r, args.metric)
        if score is not None:
            runs_with_scores.append((r, score))

    if not runs_with_scores:
        print(
            f"No finished runs with metric '{args.metric}' found in sweep "
            f"'{args.sweep_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sort by best metric
    runs_with_scores.sort(key=lambda x: x[1], reverse=True)
    runs = [r for r, _ in runs_with_scores]
    scores = {r.id: s for r, s in runs_with_scores}

    for run in runs[: args.top_n]:
        run.tags.append("top_k")
        run.update()

    # Build parameter columns from the sweep YAML when provided
    if args.sweep_config:
        swept_params, fixed_cli = _parse_sweep_yaml(args.sweep_config)
    else:
        swept_params, fixed_cli = [], {}

    # Swept params first (live values from wandb config), then fixed CLI params
    all_params = swept_params + [k for k in fixed_cli if k not in swept_params]

    # Collect rows for the top-N runs
    top_n = min(args.top_n, len(runs))
    rows = []
    for run in runs[:top_n]:
        cfg = run.config
        row = {p: _resolve_param(cfg, p, fixed_cli) for p in all_params}
        row["_name"] = run.name
        row["_score"] = scores[run.id]
        rows.append(row)

    # Dynamic column widths
    score_label = args.metric.split("/")[-1]
    name_w = max(len("Run name"), max(len(r["_name"]) for r in rows))
    score_w = max(len(score_label), 8)
    param_widths = {p: max(len(p), max(len(r[p]) for r in rows)) for p in all_params}

    # Table header
    print(f"\nTop {top_n} run(s) from sweep: {args.sweep_id}")
    print(f"Ranked by: {args.metric} (descending)\n")

    header = f"{'Rank':<6} {'Run name':<{name_w}}"
    for p in all_params:
        header += f"  {p:<{param_widths[p]}}"
    header += f"  {score_label:<{score_w}}"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(rows, start=1):
        line = f"{rank:<6} {row['_name']:<{name_w}}"
        for p in all_params:
            line += f"  {row[p]:<{param_widths[p]}}"
        line += f"  {row['_score']:<{score_w}}"
        print(line)

    # Best run detail block
    best = runs[0]
    best_cfg = best.config
    col_w = max((len(p) for p in all_params), default=0)
    col_w = max(col_w, len(score_label), len("run_name"), len("wandb_url"))
    print("\n--- Best run details ---")
    print(f"  {'run_name':<{col_w}} : {best.name}")
    for p in all_params:
        source = "wandb" if p in best_cfg else "command"
        val = _resolve_param(best_cfg, p, fixed_cli)
        print(f"  {p:<{col_w}} : {val}  [{source}]")
    print(f"  {score_label:<{col_w}} : {scores[best.id]:.4f}")
    print(f"  {'wandb_url':<{col_w}} : {best.url}")


if __name__ == "__main__":
    main()
