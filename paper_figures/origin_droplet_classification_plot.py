import csv
import os

import originpro as op


DATA_CSV = r"E:\VsCode Project\DeepLamp_4channel\convnextv2b_cls_runs\ConvNeXtV2-B\history.csv"
OUTPUT_DIR = r"E:\VsCode Project\DeepLamp_4channel\paper_figures"
PROJECT_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_training_origin.opju")
SVG_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_training_origin.svg")
PNG_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_training_origin_preview.png")
STATUS_PATH = os.path.join(OUTPUT_DIR, "origin_classification_status.txt")

COLORS = {
    "train_loss": "#2166AC",
    "validation_loss": "#4393C3",
    "validation_accuracy": "#B2182B",
    "best": "#555555",
}


def values(rows, name, percent=False):
    scale = 100.0 if percent else 1.0
    return [float(row[name]) * scale for row in rows]


def style_plot(plot, color, width=1500):
    plot.color = color
    plot.set_cmd(f"-w {width}", "-d 0")


with open(DATA_CSV, "r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

epochs = values(rows, "epoch")
train_loss = values(rows, "train_loss")
validation_loss = values(rows, "val_loss")
validation_accuracy = values(rows, "val_acc", percent=True)
validation_f1 = values(rows, "val_f1")

# The training code selects the checkpoint by validation F1, then accuracy.
best_index = max(
    range(len(rows)),
    key=lambda index: (validation_f1[index], validation_accuracy[index]),
)
best_epoch = int(epochs[best_index])
best_accuracy = validation_accuracy[best_index]
best_f1 = validation_f1[best_index] * 100.0

# Start a clean Origin project so the OPJU contains only this figure and its data.
op.new()

wks = op.new_sheet(type="w", lname="Droplet Classification Metrics")
wks.cols = 6
wks.from_list(0, epochs, lname="Epoch", axis="X")
wks.from_list(1, train_loss, lname="Train Classification Loss", axis="Y")
wks.from_list(2, validation_loss, lname="Validation Classification Loss", axis="Y")
wks.from_list(3, validation_accuracy, lname="Validation Accuracy", units="%", axis="Y")
wks.from_list(4, [best_epoch, best_epoch], lname="Best Epoch X", axis="X")
wks.from_list(5, [0.04, 0.20], lname="Best Epoch Line", axis="Y")

# Origin's built-in Double-Y template keeps the three curves in one graph while
# preserving meaningful units for loss and accuracy.
graph = op.new_graph(template="doubley", lname="Droplet Classification Training Performance")
loss_layer = graph[0]
accuracy_layer = graph[1]

style_plot(loss_layer.add_plot(wks, coly=1, colx=0, type="line"), COLORS["train_loss"])
style_plot(loss_layer.add_plot(wks, coly=2, colx=0, type="line"), COLORS["validation_loss"])
best_plot = loss_layer.add_plot(wks, coly=5, colx=4, type="line")
style_plot(best_plot, COLORS["best"], width=750)

accuracy_plot = accuracy_layer.add_plot(wks, coly=3, colx=0, type="line")
style_plot(accuracy_plot, COLORS["validation_accuracy"])

graph.set_int("aa", 1)
for layer in (loss_layer, accuracy_layer):
    layer.set_xlim(0, 30, 5)
    layer.set_int("x.showgrids", 0)
    layer.set_int("y.showgrids", 0)

loss_layer.set_ylim(0.04, 0.20, 0.02)
accuracy_layer.set_ylim(94, 100, 1)
loss_layer.axis("x").title = "Epoch"
loss_layer.axis("y").title = "Classification Loss"
accuracy_layer.axis("y").title = "Validation Accuracy (%)"

# Separate layer-aware legend boxes make every sample editable in Origin.
loss_legend = loss_layer.label("Legend")
loss_legend.text = (
    "\\l(1) Train Loss     \\l(2) Validation Loss\n"
    "\\l(2.1) Validation Accuracy\n"
    f"\\l(3) Best epoch = {best_epoch}; Accuracy = {best_accuracy:.2f}%"
)
loss_legend.set_int("fsize", 14)
loss_legend.set_int("showframe", 1)
loss_legend.set_int("left", 3000)
loss_legend.set_int("top", 650)

op.save(PROJECT_PATH)
graph.save_fig(SVG_PATH, type="svg", replace=True)
graph.save_fig(PNG_PATH, type="png", replace=True, width=2200)

with open(STATUS_PATH, "w", encoding="utf-8") as stream:
    stream.write(
        "success\n"
        f"best_epoch={best_epoch}\n"
        f"best_val_accuracy={best_accuracy:.5f}\n"
        f"best_val_f1={best_f1:.5f}\n"
    )

print(f"Saved Origin project: {PROJECT_PATH}")
print(f"Saved Origin SVG: {SVG_PATH}")
print(f"Best epoch: {best_epoch}; validation accuracy: {best_accuracy:.2f}%")
