"""汇总多个 run 的 summary.json，便于消融对比。"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison.csv"))
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        rows.append({
            "run": run.name,
            "best_epoch": summary["best_epoch"],
            "best_miou": summary["best_miou"],
            "best_dice": summary["best_dice"],
            "best_val_loss": summary["best_val_loss"],
        })
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
