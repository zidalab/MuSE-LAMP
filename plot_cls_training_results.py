"""
Plot ConvNeXt V2-B classification training results.

This script reads the files saved by train_ConvNeXtV2-B_cls.py:
  - history.csv
  - val_confusion_matrix.csv

It generates:
  - classification_loss_curve.png
  - validation_confusion_matrix.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path("convnextv2b_cls_runs") / "ConvNeXtV2-B"


def read_history(history_path):
    rows = []
    with history_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "train_f1": float(row["train_f1"]),
                "val_f1": float(row["val_f1"]),
            })
    if not rows:
        raise ValueError(f"No training rows found in {history_path}")
    return rows


def read_confusion_matrix(matrix_path):
    with matrix_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3 or len(rows[0]) < 3:
        raise ValueError(f"Unexpected confusion matrix format: {matrix_path}")

    pred_labels = [label.replace("pred_", "") for label in rows[0][1:3]]
    true_labels = [rows[1][0].replace("true_", ""), rows[2][0].replace("true_", "")]
    matrix = [
        [int(rows[1][1]), int(rows[1][2])],
        [int(rows[2][1]), int(rows[2][2])],
    ]
    return matrix, true_labels, pred_labels


def plot_loss_curve(history, output_path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    best_val_idx = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    best_epoch = history[best_val_idx]["epoch"]
    best_val_loss = history[best_val_idx]["val_loss"]

    plt.figure(figsize=(7.2, 4.8), dpi=300)
    plt.plot(epochs, train_loss, color="#2F6B9A", linewidth=2.0, label="Training loss")
    plt.plot(epochs, val_loss, color="#C7522A", linewidth=2.0, label="Validation loss")
    plt.scatter([best_epoch], [best_val_loss], color="#C7522A", s=28, zorder=3)
    plt.annotate(
        f"min val loss\nEpoch {best_epoch}",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + 1, best_val_loss + 0.01),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#555555"},
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt V2-B Classification Loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix, true_labels, pred_labels, output_path, class_names=None):
    labels = class_names or true_labels
    total = sum(sum(row) for row in matrix)
    correct = matrix[0][0] + matrix[1][1]
    accuracy = correct / total if total else 0.0

    row_sums = [sum(row) for row in matrix]
    normalized = [
        [value / row_sums[row_idx] if row_sums[row_idx] else 0.0 for value in row]
        for row_idx, row in enumerate(matrix)
    ]

    plt.figure(figsize=(5.4, 4.8), dpi=300)
    image = plt.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Row-normalized proportion")

    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            percent = normalized[i][j] * 100
            text_color = "white" if normalized[i][j] >= 0.55 else "#222222"
            plt.text(
                j,
                i,
                f"{value}\n{percent:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Validation Confusion Matrix (Accuracy={accuracy:.2%})")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_metric_summary(history, output_path):
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7.2, 4.8), dpi=300)
    plt.plot(epochs, [row["train_acc"] for row in history], linewidth=1.8, label="Training accuracy")
    plt.plot(epochs, [row["val_acc"] for row in history], linewidth=1.8, label="Validation accuracy")
    plt.plot(epochs, [row["train_f1"] for row in history], linewidth=1.8, label="Training F1")
    plt.plot(epochs, [row["val_f1"] for row in history], linewidth=1.8, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.90, 1.005)
    plt.title("ConvNeXt V2-B Classification Metrics")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot classification training curves and confusion matrix.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Training output directory.")
    parser.add_argument("--history", type=Path, default=None, help="Path to history.csv.")
    parser.add_argument("--matrix", type=Path, default=None, help="Path to val_confusion_matrix.csv.")
    parser.add_argument(
        "--class-names",
        nargs=2,
        default=["negative", "positive"],
        help="Display names for class 0 and class 1.",
    )
    parser.add_argument("--no-metrics", action="store_true", help="Skip accuracy/F1 summary plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    history_path = args.history or run_dir / "history.csv"
    matrix_path = args.matrix or run_dir / "val_confusion_matrix.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix file: {matrix_path}")

    run_dir.mkdir(parents=True, exist_ok=True)

    history = read_history(history_path)
    matrix, true_labels, pred_labels = read_confusion_matrix(matrix_path)

    loss_output = run_dir / "classification_loss_curve.png"
    matrix_output = run_dir / "validation_confusion_matrix.png"
    metrics_output = run_dir / "classification_metric_curve.png"

    plot_loss_curve(history, loss_output)
    plot_confusion_matrix(matrix, true_labels, pred_labels, matrix_output, class_names=args.class_names)
    if not args.no_metrics:
        plot_metric_summary(history, metrics_output)

    print(f"Saved loss curve: {loss_output}")
    print(f"Saved confusion matrix: {matrix_output}")
    if not args.no_metrics:
        print(f"Saved metric curve: {metrics_output}")


if __name__ == "__main__":
    main()
