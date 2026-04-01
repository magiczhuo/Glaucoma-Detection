#!/usr/bin/env python
"""Plot confusion matrices for GRINLAB prediction Excel files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_predictions(path: Path) -> pd.DataFrame:
    """Read Excel and drop unnamed index-like columns."""
    df = pd.read_excel(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df


def compute_confusion(
    df: pd.DataFrame, gt_col: str = "gt", pred_col: str = "pred_label"
) -> Tuple[List[str], np.ndarray]:
    """Build confusion matrix from ground truth and predicted labels."""
    y_true = df[gt_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    labels: List[str] = sorted({str(x) for x in np.unique(y_true)} | {str(x) for x in np.unique(y_pred)})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[label_to_idx[str(t)], label_to_idx[str(p)]] += 1
    return labels, mat


def build_annotations(mat: np.ndarray) -> np.ndarray:
    """Return array of strings with count and row percentage for each cell."""
    row_sums = mat.sum(axis=1, keepdims=True)
    total = mat.sum()
    ann = np.empty_like(mat, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            count = mat[i, j]
            row_total = row_sums[i, 0]
            row_pct = 100 * count / row_total if row_total else 0.0
            total_pct = 100 * count / total if total else 0.0
            ann[i, j] = f"{count}\n{row_pct:.1f}% row\n{total_pct:.1f}% all"
    return ann


def plot_confusion(
    labels: Sequence[str],
    mat: np.ndarray,
    title: str,
    outfile: Path,
    cmap: str = "Blues",
    dpi: int = 300,
) -> None:
    """Plot row-normalized confusion matrix with count + percentage annotations."""
    row_norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    ann = build_annotations(mat)

    fig, ax = plt.subplots(figsize=(4.5 + 0.6 * len(labels), 4.2 + 0.4 * len(labels)))
    sns.heatmap(
        row_norm,
        annot=ann,
        fmt="",
        cmap=cmap,
        cbar_kws={"label": "Row-normalized rate"},
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    acc = mat.trace() / mat.sum() if mat.sum() else 0.0
    ax.set_title(f"{title}\nAccuracy = {acc*100:.2f}% (n={mat.sum()})", pad=14)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def default_jobs() -> Iterable[Tuple[str, Path]]:
    base = Path("results")
    return [
        ("3b-rcbam-f12-b-best-2", base / "3b-rcbam-f12-b-best-2" / "prediction.xlsx"),
        ("3b-rcbam-f12-3cls-best", base / "3b-rcbam-f12-3cls-best" / "prediction.xlsx"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrices from prediction Excel files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/confusion_matrices"),
        help="Directory to save confusion matrix PNG files.",
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help="Matplotlib colormap name (e.g., Blues, PuBu, OrRd).",
    )
    args = parser.parse_args()

    for name, path in default_jobs():
        if not path.exists():
            print(f"[skip] Missing file: {path}")
            continue
        df = load_predictions(path)
        labels, mat = compute_confusion(df)
        outfile = args.output_dir / f"confmat_{name}.png"
        plot_confusion(labels, mat, f"{name}", outfile, cmap=args.cmap)
        print(f"[done] Saved {outfile}")


if __name__ == "__main__":
    main()
