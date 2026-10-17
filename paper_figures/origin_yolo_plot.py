import csv
import os

import originpro as op


DATA_CSV = r"E:\VsCode Project\DeepLamp_4channel\runs\segment\seg_runs\droplet_seg_v2\results.csv"
OUTPUT_DIR = r"E:\VsCode Project\DeepLamp_4channel\paper_figures"
PROJECT_PATH = os.path.join(OUTPUT_DIR, "yolo_training_curves_origin.opju")
SVG_PATH = os.path.join(OUTPUT_DIR, "yolo_training_curves_origin.svg")
PNG_PATH = os.path.join(OUTPUT_DIR, "yolo_training_curves_origin_preview.png")
STATUS_PATH = os.path.join(OUTPUT_DIR, "origin_yolo_status.txt")

COLORS = {
    "box_train": "#2166AC",
    "box_validation": "#4393C3",
    "mask_train": "#B2182B",
    "mask_validation": "#D6604D",
    "best": "#555555",
}


def values(rows, name, percent=False):
    scale = 100.0 if percent else 1.0
    return [float(row[name]) * scale for row in rows]


with open(DATA_CSV, "r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

epochs = values(rows, "epoch")
train_box = values(rows, "train/box_loss")
validation_box = values(rows, "val/box_loss")
train_mask = values(rows, "train/seg_loss")
validation_mask = values(rows, "val/seg_loss")
box_map50 = values(rows, "metrics/mAP50(B)", percent=True)
box_map5095 = values(rows, "metrics/mAP50-95(B)", percent=True)
mask_map50 = values(rows, "metrics/mAP50(M)", percent=True)
mask_map5095 = values(rows, "metrics/mAP50-95(M)", percent=True)

best_index = max(range(len(rows)), key=lambda index: mask_map5095[index])
best_epoch = int(epochs[best_index])
best_mask_map5095 = mask_map5095[best_index]

# Remove incomplete pages left by a previous interrupted run.
generated_page_names = {
    "YOLO Training Metrics",
    "Box and Mask Loss",
    "Box and Mask mAP",
    "YOLO Training Curves",
}
for page_type in ("g", "w"):
    for page in list(op.pages(page_type)):
        if page.lname in generated_page_names:
            page.destroy()

# Create a native Origin worksheet with explicit X/Y designations.
wks = op.new_sheet(type="w", lname="YOLO Training Metrics")
wks.cols = 13
wks.from_list(0, epochs, lname="Epoch", axis="X")
wks.from_list(1, train_box, lname="Train Box Loss", axis="Y")
wks.from_list(2, validation_box, lname="Validation Box Loss", axis="Y")
wks.from_list(3, train_mask, lname="Train Mask Loss", axis="Y")
wks.from_list(4, validation_mask, lname="Validation Mask Loss", axis="Y")
wks.from_list(5, box_map50, lname="Box mAP@0.5", units="%", axis="Y")
wks.from_list(6, box_map5095, lname="Box mAP@0.5:0.95", units="%", axis="Y")
wks.from_list(7, mask_map50, lname="Mask mAP@0.5", units="%", axis="Y")
wks.from_list(8, mask_map5095, lname="Mask mAP@0.5:0.95", units="%", axis="Y")
wks.from_list(9, [best_epoch, best_epoch], lname="Loss Best Epoch X", axis="X")
wks.from_list(10, [0, 1.8], lname="Loss Best Epoch Line", axis="Y")
wks.from_list(11, [best_epoch, best_epoch], lname="mAP Best Epoch X", axis="X")
wks.from_list(12, [40, 100], lname="mAP Best Epoch Line", axis="Y")


def format_common(graph, layer, y_title, y_limits, y_step):
    graph.set_int("aa", 1)
    layer.set_xlim(0, 70, 10)
    layer.set_ylim(y_limits[0], y_limits[1], y_step)
    layer.axis("x").title = "Epoch"
    layer.axis("y").title = y_title
    layer.set_int("x.showgrids", 0)
    layer.set_int("y.showgrids", 0)
    layer.set_int("x.opposite", 0)
    layer.set_int("y.opposite", 0)


def style_plot(plot, color):
    plot.color = color
    # Origin stores line width in 1/500 point units; 1500 is 3 pt.
    plot.set_cmd("-w 1500", "-d 0")


# Panel (a): native four-curve loss graph.
loss_graph = op.new_graph(template="line", lname="Box and Mask Loss")
loss_layer = loss_graph[0]
loss_specs = [
    (1, COLORS["box_train"]),
    (2, COLORS["box_validation"]),
    (3, COLORS["mask_train"]),
    (4, COLORS["mask_validation"]),
]
for column, color in loss_specs:
    style_plot(loss_layer.add_plot(wks, coly=column, colx=0, type="line"), color)
loss_best_plot = loss_layer.add_plot(wks, coly=10, colx=9, type="line")
loss_best_plot.color = COLORS["best"]
loss_best_plot.set_cmd("-w 750", "-d 0")
format_common(loss_graph, loss_layer, "Loss", (0, 1.8), 0.3)

loss_legend = loss_layer.label("Legend")
loss_legend.text = (
    "\\l(1) Train Box     \\l(2) Validation Box\n"
    "\\l(3) Train Mask    \\l(4) Validation Mask\n"
    f"\\l(5) Best epoch = {best_epoch}"
)
loss_legend.set_int("fsize", 18)
loss_legend.set_int("showframe", 1)
loss_legend.set_int("left", 2350)
loss_legend.set_int("top", 700)

# Panel (b): native four-curve box/mask mAP graph.
map_graph = op.new_graph(template="line", lname="Box and Mask mAP")
map_layer = map_graph[0]
map_specs = [
    (5, COLORS["box_train"]),
    (6, COLORS["box_validation"]),
    (7, COLORS["mask_train"]),
    (8, COLORS["mask_validation"]),
]
for column, color in map_specs:
    style_plot(map_layer.add_plot(wks, coly=column, colx=0, type="line"), color)
map_best_plot = map_layer.add_plot(wks, coly=12, colx=11, type="line")
map_best_plot.color = COLORS["best"]
map_best_plot.set_cmd("-w 750", "-d 0")
format_common(map_graph, map_layer, "mAP (%)", (40, 100), 10)

map_legend = map_layer.label("Legend")
map_legend.text = (
    "\\l(1) Box mAP@0.5     \\l(2) Box mAP@0.5:0.95\n"
    "\\l(3) Mask mAP@0.5    \\l(4) Mask mAP@0.5:0.95\n"
    f"\\l(5) Best epoch = {best_epoch}; Mask mAP@0.5:0.95 = {best_mask_map5095:.2f}%"
)
map_legend.set_int("fsize", 14)
map_legend.set_int("showframe", 1)
map_legend.set_int("left", 1350)
map_legend.set_int("top", 3500)

# Merge the two native graph pages into a compact one-row Origin graph.
loss_name = loss_graph.name
map_name = map_graph.name
merge_command = (
    f'merge_graph option:=specified graphs:="{loss_name}"+char(10)$+"{map_name}" '
    "keep:=0 row:=1 col:=2 aspectratio:=0 xgap:=2 leftmg:=3 rightmg:=3 "
    "topmg:=5 bottommg:=5 width:=6.4 height:=5.6 unit:=cm scalemode:=fixed fixfactor:=0.90 "
    "labeltext:=alphabet "
    "labelposition:=tlo;"
)
op.lt_exec(merge_command)

merged = op.find_graph()
merged.lname = "YOLO Training Curves"
merged.set_int("aa", 1)

# Origin project is the primary editable deliverable; SVG is exported by Origin.
op.save(PROJECT_PATH)
merged.save_fig(SVG_PATH, type="svg", replace=True)
merged.save_fig(PNG_PATH, type="png", replace=True, width=2700)

with open(STATUS_PATH, "w", encoding="utf-8") as stream:
    stream.write(f"success\nbest_epoch={best_epoch}\nmask_map5095={best_mask_map5095:.5f}\n")

print(f"Saved Origin project: {PROJECT_PATH}")
print(f"Saved Origin SVG: {SVG_PATH}")
