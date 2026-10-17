import csv
import os

import originpro as op


DATA_CSV = r"E:\VsCode Project\DeepLamp_4channel\convnextv2b_cls_runs\ConvNeXtV2-B\val_confusion_matrix.csv"
OUTPUT_DIR = r"E:\VsCode Project\DeepLamp_4channel\paper_figures"
PROJECT_PATH = os.path.join(
    OUTPUT_DIR, "droplet_classification_normalized_confusion_matrix_origin.opju"
)
SVG_PATH = os.path.join(
    OUTPUT_DIR, "droplet_classification_normalized_confusion_matrix_origin.svg"
)
PNG_PATH = os.path.join(
    OUTPUT_DIR, "droplet_classification_normalized_confusion_matrix_origin_preview.png"
)
STATUS_PATH = os.path.join(
    OUTPUT_DIR, "origin_normalized_confusion_matrix_status.txt"
)
PALETTE_PATH = r"D:\OriginLab\Palettes\Lite LT Red.pal"

CLASS_NAMES = ["Negative", "Positive"]


with open(DATA_CSV, "r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.reader(stream))

raw_matrix = [
    [int(rows[1][1]), int(rows[1][2])],
    [int(rows[2][1]), int(rows[2][2])],
]

# Standard confusion-matrix normalization: divide every cell by the total of
# its true-class row. Each row therefore sums to 1.
row_totals = [sum(row) for row in raw_matrix]
matrix = [
    [value / row_total for value in row]
    for row, row_total in zip(raw_matrix, row_totals)
]

total = sum(row_totals)
correct = raw_matrix[0][0] + raw_matrix[1][1]
accuracy = correct / total

# Origin maps the first matrix row to the lower Y coordinate. Flip the source
# rows so the conventional order remains Negative above Positive.
matrix_for_origin = list(reversed(matrix))

op.lt_exec('system.font.default$="Arial"; system.font.labelName$="Arial";')
op.new()

mat = op.new_sheet(type="m", lname="Validation Normalized Confusion Matrix")
mat.shape = (2, 2)
if not mat.obj.MatrixObjects(0).SetData(matrix_for_origin, 0, 0):
    raise RuntimeError("Failed to write the normalized confusion matrix into Origin")
mat.set_label(0, "Proportion")
mat.xymap = (1, 2, 1, 2)

graph = op.new_graph(
    template="Heat_Map.otpu",
    lname="Droplet Classification Normalized Validation Confusion Matrix",
)
layer = graph[0]
plot = layer.add_mplot(mat, 0, type=105)

# A common 0-to-1 scale allows direct comparison between all cells/classes.
zlevels = plot.zlevels
zlevels["minors"] = 64
zlevels["levels"] = [0, 1]
plot.zlevels = zlevels
layer.activate()
op.lt_exec(
    f'palApply fname:="{PALETTE_PATH}" flip:=1 stretch:=1 scope:=all;'
)

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

layer.activate()
op.lt_exec(
    "layer.unit=1; layer.left=19; layer.top=10; "
    "layer.width=57; layer.height=67; "
    "layer.x.label.font=font(Arial); layer.y.label.font=font(Arial); "
    "xb.font=font(Arial); yl.font=font(Arial); xb.fsize=20; yl.fsize=20;"
)

# Show three decimals. High-proportion diagonal cells use white text, while
# low-proportion off-diagonal cells use dark text for contrast.
cell_labels = [
    ("CM11", f"{matrix[0][0]:.3f}", 25, 25, "#FFFFFF"),
    ("CM12", f"{matrix[0][1]:.3f}", 75, 25, "#222222"),
    ("CM21", f"{matrix[1][0]:.3f}", 25, 75, "#222222"),
    ("CM22", f"{matrix[1][1]:.3f}", 75, 75, "#FFFFFF"),
]
layer.activate()
for name, text, x_percent, y_percent, color in cell_labels:
    op.lt_exec(f'label -p {x_percent} {y_percent} -j 1 -n {name} {text};')
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
        f"raw_matrix={raw_matrix}\n"
        f"row_totals={row_totals}\n"
        f"normalized_matrix={matrix}\n"
        f"accuracy={accuracy:.8f}\n"
    )

print(f"Saved normalized Origin project: {PROJECT_PATH}")
print(f"Saved normalized SVG: {SVG_PATH}")
print(f"Normalized confusion matrix: {matrix}")
print(f"Validation accuracy: {accuracy:.4f}")
