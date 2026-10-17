import csv
import os

import originpro as op


DATA_CSV = r"E:\VsCode Project\DeepLamp_4channel\convnextv2b_cls_runs\ConvNeXtV2-B\val_confusion_matrix.csv"
OUTPUT_DIR = r"E:\VsCode Project\DeepLamp_4channel\paper_figures"
PROJECT_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_confusion_matrix_origin.opju")
SVG_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_confusion_matrix_origin.svg")
PNG_PATH = os.path.join(OUTPUT_DIR, "droplet_classification_confusion_matrix_origin_preview.png")
STATUS_PATH = os.path.join(OUTPUT_DIR, "origin_confusion_matrix_status.txt")
PALETTE_PATH = r"D:\OriginLab\Palettes\Lite LT Red.pal"

CLASS_NAMES = ["Negative", "Positive"]


with open(DATA_CSV, "r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.reader(stream))

matrix = [
    [int(rows[1][1]), int(rows[1][2])],
    [int(rows[2][1]), int(rows[2][2])],
]

total = sum(sum(row) for row in matrix)
correct = matrix[0][0] + matrix[1][1]
accuracy = correct / total * 100.0

# Origin maps the first matrix row to the lower Y coordinate. Flip the source
# rows so the conventional confusion-matrix order remains Negative above Positive.
matrix_for_origin = list(reversed(matrix))

# Use a clean sans-serif typeface throughout the newly created Origin graph.
# These session-level settings do not overwrite Origin's startup preferences.
op.lt_exec('system.font.default$="Arial"; system.font.labelName$="Arial";')
op.new()

mat = op.new_sheet(type="m", lname="Validation Confusion Matrix Counts")
mat.shape = (2, 2)
if not mat.obj.MatrixObjects(0).SetData(matrix_for_origin, 0, 0):
    raise RuntimeError("Failed to write the confusion matrix into Origin")
mat.set_label(0, "Count")
mat.xymap = (1, 2, 1, 2)

graph = op.new_graph(
    template="Heat_Map.otpu",
    lname="Droplet Classification Validation Confusion Matrix",
)
layer = graph[0]
plot = layer.add_mplot(mat, 0, type=105)

# Smooth white-to-deep-red scale, matching the supplied reference style.
zlevels = plot.zlevels
zlevels["minors"] = 64
zlevels["levels"] = [0, max(max(row) for row in matrix)]
plot.zlevels = zlevels
layer.activate()
op.lt_exec(
    f'palApply fname:="{PALETTE_PATH}" flip:=1 stretch:=1 scope:=all;'
)

# Square cells, class names at cell centers, and a clean four-sided frame.
layer.set_xlim(0.5, 2.5, 1)
layer.set_ylim(0.5, 2.5, 1)
layer.set_str("x.ticksbydata", "1 2")
layer.set_str("y.ticksbydata", "1 2")
layer.set_int("x.label.type", 10)
layer.set_str("x.label.string", "Negative Positive")
layer.set_int("y.label.type", 10)
layer.set_str("y.label.string", "Positive Negative")
layer.set_int("x.label.pt", 18)
layer.set_int("y.label.pt", 18)
layer.set_int("x.label.rotate", 0)
layer.set_int("y.label.rotate", 0)
layer.set_int("x.showgrids", 0)
layer.set_int("y.showgrids", 0)
layer.set_int("x.opposite", 1)
layer.set_int("y.opposite", 1)
layer.axis("x").title = "Predicted class"
layer.axis("y").title = "True class"

# Reserve enough left margin for the vertical title and keep the heatmap
# square on the exported landscape page.
layer.activate()
op.lt_exec(
    "layer.unit=1; layer.left=19; layer.top=10; "
    "layer.width=57; layer.height=67; "
    "layer.x.label.font=font(Arial); layer.y.label.font=font(Arial); "
    "xb.font=font(Arial); yl.font=font(Arial); xb.fsize=20; yl.fsize=20;"
)

# Cell values are native editable Origin text objects. The two high-count
# diagonal cells use white text; low-count off-diagonal cells use dark text.
cell_labels = [
    ("CM11", str(matrix[0][0]), 25, 25, "#FFFFFF"),
    ("CM12", str(matrix[0][1]), 75, 25, "#222222"),
    ("CM21", str(matrix[1][0]), 25, 75, "#222222"),
    ("CM22", str(matrix[1][1]), 75, 75, "#FFFFFF"),
]
layer.activate()
for name, text, x_percent, y_percent, color in cell_labels:
    op.lt_exec(f'label -p {x_percent} {y_percent} -j 1 -n {name} {text};')
    # Keep these assignments separate: this Origin build does not expose a
    # ``bold`` graphic-object property, and one invalid assignment would stop
    # the remaining style assignments on the same LabTalk line.
    op.lt_exec(f'{name}.anchor=5;')
    op.lt_exec(f'{name}.fsize=24;')
    op.lt_exec(f'{name}.font=font(Arial);')
    op.lt_exec(f'{name}.color=color("{color}");')

op.save(PROJECT_PATH)
graph.save_fig(SVG_PATH, type="svg", replace=True)
graph.save_fig(PNG_PATH, type="png", replace=True, width=2000)

with open(STATUS_PATH, "w", encoding="utf-8") as stream:
    stream.write(
        "success\n"
        f"matrix={matrix}\n"
        f"total={total}\n"
        f"correct={correct}\n"
        f"accuracy={accuracy:.5f}\n"
    )

print(f"Saved Origin project: {PROJECT_PATH}")
print(f"Saved Origin SVG: {SVG_PATH}")
print(f"Validation confusion matrix accuracy: {accuracy:.2f}%")
