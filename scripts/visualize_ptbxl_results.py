#!/usr/bin/env python3
"""Create small PTB-XL result visualizations from HeartLang evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


TASK_LABELS = {
    "PTBXL_QRS_superdiagnostic": ["NORM", "MI", "STTC", "CD", "HYP"],
}


def read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def label_names(dataset_name: str, width: int) -> list[str]:
    labels = TASK_LABELS.get(dataset_name)
    if labels and len(labels) == width:
        return labels
    return [f"label_{idx}" for idx in range(width)]


def plot_auc_summary(results_csv: Path, output_dir: Path) -> Path | None:
    if not results_csv.exists():
        return None

    results = pd.read_csv(results_csv)
    if results.empty or "ROC AUC" not in results.columns:
        return None

    results["Run"] = (
        results["Dataset Directory"].astype(str).str.split("/").str[-1]
        + " | "
        + results["Trainable Layer"].astype(str)
        + " | "
        + results["Split Ratio"].astype(str)
    )

    fig_width = max(8, min(16, len(results) * 1.4))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(results["Run"], results["ROC AUC"], color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro ROC AUC")
    ax.set_title("HeartLang PTB-XL Evaluation Summary")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    output_path = output_dir / "auc_summary.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def summarize_prediction_pair(outputs_path: Path, targets_path: Path, output_dir: Path) -> pd.DataFrame:
    dataset_name = outputs_path.name.removesuffix("_outputs.csv")
    outputs = read_matrix(outputs_path)
    targets = read_matrix(targets_path)

    if outputs.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch for {dataset_name}: outputs={outputs.shape}, targets={targets.shape}"
        )

    names = label_names(dataset_name, outputs.shape[1])
    rows = []

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for idx, name in enumerate(names):
        y_true = targets[:, idx]
        y_score = outputs[:, idx]

        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)
        if positives == 0 or negatives == 0:
            auc = np.nan
            ap = np.nan
            continue

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} AUC={auc:.3f}")
        rows.append(
            {
                "dataset": dataset_name,
                "label": name,
                "positive_count": positives,
                "negative_count": negatives,
                "roc_auc": auc,
                "average_precision": ap,
            }
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-label ROC Curves: {dataset_name}")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / f"{dataset_name}_roc_curves.png"
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)

    positives = outputs[targets == 1]
    negatives = outputs[targets == 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(negatives, bins=40, alpha=0.65, label="negative labels", color="#94a3b8")
    ax.hist(positives, bins=40, alpha=0.65, label="positive labels", color="#ef4444")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Score Distribution: {dataset_name}")
    ax.legend()
    fig.tight_layout()
    hist_path = output_dir / f"{dataset_name}_score_distribution.png"
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heartlang-dir",
        type=Path,
        default=Path("external/HeartLang"),
        help="Path to the upstream HeartLang checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/figures"),
        help="Directory for generated figures and metrics summaries.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = args.heartlang_dir / "results" / "finetune_results.csv"
    pred_dir = args.heartlang_dir / "results" / "pred"

    created = []
    summary_path = plot_auc_summary(results_csv, args.output_dir)
    if summary_path:
        created.append(summary_path)

    metric_frames = []
    if pred_dir.exists():
        for outputs_path in sorted(pred_dir.glob("*_outputs.csv")):
            dataset_name = outputs_path.name.removesuffix("_outputs.csv")
            targets_path = pred_dir / f"{dataset_name}_targets.csv"
            if targets_path.exists():
                metric_frames.append(
                    summarize_prediction_pair(outputs_path, targets_path, args.output_dir)
                )

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics_path = args.output_dir / "per_label_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        created.append(metrics_path)

    if not created and not metric_frames:
        print("No HeartLang result CSV files found yet. Run evaluation first.")
    else:
        print(f"Saved visualizations and summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()
