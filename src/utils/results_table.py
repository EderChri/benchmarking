"""
Usage:
    python src/utils/results_table.py 4 5 6 7
    python src/utils/results_table.py 4 5 6 --metrics rmse mae
    python src/utils/results_table.py 4 5 6 --csv results.csv
"""

import argparse
import os
import sys
import yaml


INDEX_PATH = os.path.join(os.path.dirname(__file__), "../results/index.yaml")


def load_index():
    with open(INDEX_PATH) as f:
        return yaml.safe_load(f) or {}


def load_result(run_dir):
    path = os.path.join(os.path.dirname(__file__), "../..", run_dir, "result.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_results(results: dict, prefix="") -> dict:
    """Recursively flatten nested metric dicts (e.g. auroc_auprc: {auroc: x, auprc: y})."""
    flat = {}
    for k, v in results.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_results(v, prefix=key))
        else:
            flat[key] = v
    return flat


def build_table(run_ids: list, metric_filter: list):
    index = load_index()
    rows = []

    for run_id in run_ids:
        if run_id not in index:
            print(f"Warning: run {run_id} not found in index, skipping.", file=sys.stderr)
            continue
        result = load_result(index[run_id])
        flat_metrics = flatten_results(result.get("results") or {})

        if metric_filter:
            flat_metrics = {k: v for k, v in flat_metrics.items() if k in metric_filter}

        rows.append({
            "id": run_id,
            "name": result.get("experiment", ""),
            "status": result.get("status", ""),
            "model": result.get("run_cfg", {}).get("model", ""),
            "data": result.get("run_cfg", {}).get("data", ""),
            **flat_metrics,
        })

    return rows


def format_value(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else ""


def print_table(rows: list[dict]):
    if not rows:
        print("No results to display.")
        return

    # Collect all columns, keeping id/name/status/model/data first
    fixed = ["id", "name", "status", "model", "data"]
    metric_cols = sorted({k for row in rows for k in row if k not in fixed})
    cols = fixed + metric_cols

    # Compute column widths
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


def write_csv(rows: list[dict], path: str):
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
    parser.add_argument("runs", nargs="+", type=int, help="Run IDs to compare")
    parser.add_argument("--metrics", nargs="+", help="Only show these metric columns")
    parser.add_argument("--csv", metavar="FILE", help="Also write output to a CSV file")
    args = parser.parse_args()

    rows = build_table(args.runs, args.metrics)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
