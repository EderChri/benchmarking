"""
Usage:
    # From index (existing behavior)
    python src/utils/results_table.py 20 27 28 29
    python src/utils/results_table.py 20 27 28 29 --metrics rmse mae

    # From log files — show all runs, filter to finetune only
    python src/utils/results_table.py --logs src/results/logs/*.log --finetune-only

    # Mix: explicit run IDs, supplemented with log data
    python src/utils/results_table.py 20 27 28 --logs src/results/logs/*.log

    # Export to CSV
    python src/utils/results_table.py --logs src/results/logs/*.log --finetune-only --csv out.csv
"""

import argparse
import os
import re
import sys
import yaml


INDEX_PATH = os.path.join(os.path.dirname(__file__), "../results/index.yaml")
EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), "../../conf/experiments.yaml")

# Datasets used only for pretraining (source domain); runs on these without a
# pretrained_run are treated as pretrain-only and excluded by --finetune-only.
_PRETRAIN_DATASETS = {"ettm1", "etth1"}

_RE_RUN_HEADER = re.compile(r'\[Run (\d+)\] (.+?) \u2014 (.+)')
_RE_EPOCH = re.compile(r'Epoch (\d+): train_loss=([\d.]+)(?:, train_loss_c=[\d.]+)?, val_loss=([\d.]+)')
_RE_METRIC = re.compile(r'\[INFO\] ([A-Z][A-Z0-9_]+): ([\d.]+)')


def load_index():
    if not os.path.exists(INDEX_PATH):
        return {}
    with open(INDEX_PATH) as f:
        return yaml.safe_load(f) or {}


def load_result(run_dir):
    path = os.path.join(os.path.dirname(__file__), "../..", run_dir, "result.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiments():
    if not os.path.exists(EXPERIMENTS_PATH):
        return {}
    with open(EXPERIMENTS_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return {r["id"]: r for r in (cfg.get("runs") or []) if "id" in r}


def is_pretrain_run(run_cfg):
    """True if this run only trains on a source domain (should be excluded by --finetune-only)."""
    mode = (
        ((run_cfg.get("overrides") or {}).get("model") or {})
        .get("params", {})
        .get("mode", "")
    )
    if mode == "pretrain":
        return True
    # RawForecaster source-domain runs: no pretrained_run, no skip_training, source dataset
    if not run_cfg.get("pretrained_run") and not run_cfg.get("skip_training"):
        if run_cfg.get("data", "") in _PRETRAIN_DATASETS:
            return True
    return False


def parse_logs(log_paths):
    """Parse log files. Returns dict: run_id → run_data."""
    runs = {}
    for log_path in log_paths:
        current_id = None
        with open(log_path) as f:
            for line in f:
                m = _RE_RUN_HEADER.search(line)
                if m:
                    current_id = int(m.group(1))
                    if current_id not in runs:
                        runs[current_id] = {
                            "id": current_id,
                            "name": m.group(2).strip(),
                            "task": m.group(3).strip(),
                            "last_epoch": None,
                            "last_train_loss": None,
                            "last_val_loss": None,
                            "final_metrics": {},
                            "log_file": os.path.basename(log_path),
                            "status": "in-progress",
                        }
                    continue

                if current_id is None:
                    continue

                m = _RE_EPOCH.search(line)
                if m:
                    runs[current_id]["last_epoch"] = int(m.group(1))
                    runs[current_id]["last_train_loss"] = float(m.group(2))
                    runs[current_id]["last_val_loss"] = float(m.group(3))
                    continue

                m = _RE_METRIC.search(line)
                if m:
                    runs[current_id]["final_metrics"][m.group(1).lower()] = float(m.group(2))

    for run in runs.values():
        if run["final_metrics"]:
            run["status"] = "completed"

    return runs


def flatten_results(results, prefix=""):
    flat = {}
    for k, v in (results or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_results(v, prefix=key))
        else:
            flat[key] = v
    return flat


def build_table(run_ids, metric_filter, log_paths=None, finetune_only=False):
    index = load_index()
    experiments = load_experiments() if finetune_only else {}
    log_runs = parse_logs(log_paths) if log_paths else {}

    if not run_ids:
        run_ids = sorted(log_runs.keys())

    rows = []
    for run_id in run_ids:
        if finetune_only and run_id in experiments and is_pretrain_run(experiments[run_id]):
            continue

        row = {"id": run_id}

        # Primary source: result.yaml via index
        if run_id in index:
            try:
                result = load_result(index[run_id])
                flat = flatten_results(result.get("results"))
                if metric_filter:
                    flat = {k: v for k, v in flat.items() if k in metric_filter}
                row.update({
                    "name": result.get("experiment", ""),
                    "status": result.get("status", ""),
                    "model": (result.get("run_cfg") or {}).get("model", ""),
                    "data": (result.get("run_cfg") or {}).get("data", ""),
                    **flat,
                })
                rows.append(row)
                continue
            except (FileNotFoundError, TypeError):
                pass

        # Fall back to log data
        if run_id in log_runs:
            log = log_runs[run_id]
            exp = experiments.get(run_id) or {}
            if log["final_metrics"]:
                metrics = dict(log["final_metrics"])
            elif log["last_train_loss"] is not None:
                metrics = {
                    "train_loss": log["last_train_loss"],
                    "val_loss": log["last_val_loss"],
                    "last_epoch": log["last_epoch"],
                }
            else:
                metrics = {}
            if metric_filter:
                metrics = {k: v for k, v in metrics.items() if k in metric_filter}
            row.update({
                "name": log["name"],
                "status": log["status"],
                "model": exp.get("model", ""),
                "data": exp.get("data", ""),
                **metrics,
            })
            rows.append(row)
            continue

        print(f"Warning: run {run_id} not found in index or logs, skipping.", file=sys.stderr)

    return rows


def format_value(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else ""


def print_table(rows):
    if not rows:
        print("No results to display.")
        return

    fixed = ["id", "name", "status", "model", "data"]
    metric_cols = sorted({k for row in rows for k in row if k not in fixed})
    cols = fixed + metric_cols

    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(format_value(row.get(c))))

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"

    print(sep)
    print(header)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(format_value(row.get(c)).ljust(widths[c]) for c in cols) + " |"
        print(line)
    print(sep)


def write_csv(rows, path):
    import csv
    if not rows:
        return
    fixed = ["id", "name", "status", "model", "data"]
    metric_cols = sorted({k for row in rows for k in row if k not in fixed})
    cols = fixed + metric_cols
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare results across experiment runs.")
    parser.add_argument("runs", nargs="*", type=int, help="Run IDs to compare (optional when --logs is given)")
    parser.add_argument("--logs", nargs="+", metavar="FILE", help="Log file(s) to parse for run data")
    parser.add_argument("--finetune-only", action="store_true",
                        help="Skip pretrain/source-domain runs (reads conf/experiments.yaml)")
    parser.add_argument("--metrics", nargs="+", help="Only show these metric columns")
    parser.add_argument("--csv", metavar="FILE", help="Also write output to a CSV file")
    args = parser.parse_args()

    if not args.runs and not args.logs:
        parser.error("Provide run IDs or --logs (or both).")

    rows = build_table(args.runs, args.metrics, log_paths=args.logs, finetune_only=args.finetune_only)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
