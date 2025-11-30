#!/usr/bin/env python3
"""Plot per-method Average Accuracy curves as standalone images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_average_accuracy import collect_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-method Average Accuracy plots from PEFT-CL logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory containing per-method log folders.",
    )
    parser.add_argument(
        "--dataset-subpath",
        type=str,
        default=None,
        help="Optional subdirectory (e.g., cifar224/0/10). If omitted, scans all logs.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of method folder names to include.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to display on the x-axis.",
    )
    parser.add_argument(
        "--start-token",
        type=str,
        default="Learning on 0-10",
        help="Marker indicating the first task. Last occurrence marks the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/per_method_curves"),
        help="Directory to store the generated plots.",
    )
    parser.add_argument(
        "--img-format",
        type=str,
        default="png",
        help="Image format/extension for the saved plots (e.g., png, pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots per inch for saved plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each plot interactively (mainly for debugging).",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_single_curve(
    method: str,
    values: Iterable[float],
    max_tasks: int,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    plt.figure(figsize=(6, 4))
    x = list(range(1, len(values) + 1))
    plt.plot(x, list(values), marker="o", label=method)
    plt.xlabel("Task #")
    plt.ylabel("Average Accuracy (CNN, %)")
    plt.title(f"{method} Average Accuracy")
    plt.xticks(range(1, max_tasks + 1))
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()

    logs_dir = args.logs_dir.resolve()
    dataset_subpath = (
        Path(args.dataset_subpath.strip("/")) if args.dataset_subpath else None
    )
    methods_filter: Optional[Iterable[str]] = set(args.methods) if args.methods else None
    ensure_output_dir(args.output_dir)

    print(f"[INFO] Collecting curves from {logs_dir}")
    curves = collect_curves(
        logs_dir=logs_dir,
        dataset_subpath=dataset_subpath,
        methods_filter=methods_filter,
        start_token=args.start_token,
        max_tasks=args.max_tasks,
    )

    if not curves:
        raise SystemExit("No curves found to plot.")

    for method, (values, log_path) in sorted(curves.items()):
        output_path = args.output_dir / f"{method}.{args.img_format.lstrip('.')}"
        print(f"[INFO] Plotting {method} -> {output_path} (source: {log_path})")
        plot_single_curve(method, values, args.max_tasks, output_path, args.dpi, args.show)


if __name__ == "__main__":
    main()

