"""
YOLO 液滴实例分割训练脚本。

流程：
  1. 将 COCO RLE mask 转换为 YOLO 分割格式（归一化多边形 .txt）
  2. 随机划分 train / val
  3. 生成 dataset.yaml
  4. 启动 YOLO-seg 训练
"""

import json
import multiprocessing
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 用户配置
# ─────────────────────────────────────────────────────────────────────────────

COCO_JSON       = r"dataset_seg/_annotations.coco.json"
IMAGES_DIR      = r"dataset_seg/images"

# 转换后的 YOLO 数据集存放目录
OUTPUT_DIR      = r"dataset_seg_yolo"

# 训练参数
BASE_MODEL      = r"yolo11n-seg.pt"   # 领域专用预训练权重
EPOCHS          = 200
IMG_SIZE        = 1280
BATCH           = 16                  # 根据显存调整
VAL_RATIO       = 0.2                # 验证集比例
RANDOM_SEED     = 42

# 训练输出目录
TRAIN_PROJECT   = "seg_runs"
TRAIN_NAME      = "droplet_seg_v2"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: RLE mask → YOLO polygon txt
# ─────────────────────────────────────────────────────────────────────────────

def rle_to_mask(rle: dict) -> np.ndarray:
    """将 COCO RLE(counts 字符串)解码为二值掩膜 (H, W)。"""
    try:
        from pycocotools import mask as mask_util
        return mask_util.decode(rle).astype(np.uint8)
    except ImportError:
        pass
    # 不依赖 pycocotools 的手动实现（仅处理 uncompressed RLE list）
    if isinstance(rle["counts"], list):
        h, w = rle["size"]
        mask = np.zeros(h * w, dtype=np.uint8)
        pos, fill = 0, 0
        for count in rle["counts"]:
            if fill:
                mask[pos : pos + count] = 1
            pos += count
            fill ^= 1
        return mask.reshape((h, w), order="F")
    raise ValueError("counts 是字符串（compressed RLE），请先安装 pycocotools: pip install pycocotools")


def mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int) -> list[float] | None:
    """将二值掩膜转换为归一化的 YOLO 多边形点列表 [x1,y1,x2,y2,...]。
    取面积最大的连通域轮廓，使用 approxPolyDP 适当简化。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    # 较小的 epsilon 保留更多边界细节，作为标注精度的上限
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(float)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    pts = np.clip(pts, 0.0, 1.0)
    return pts.flatten().tolist()


def convert_dataset(
    coco_json: str,
    images_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """把 COCO 数据集转为 YOLO 分割格式，返回类别名列表和输出路径信息。"""
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    categories   = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_meta   = {img["id"]: img for img in coco["images"]}
    # category_id → yolo_class_index（按 id 排序）
    sorted_cat_ids = sorted(categories.keys())
    cat_to_idx     = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}
    class_names    = [categories[cid] for cid in sorted_cat_ids]

    # 按 image_id 收集 annotations
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # 划分 train / val
    image_ids = list(id_to_meta.keys())
    random.seed(seed)
    random.shuffle(image_ids)
    n_val   = max(1, int(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:n_val])

    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for img_id, meta in id_to_meta.items():
        split    = "val" if img_id in val_ids else "train"
        src_img  = Path(images_dir) / meta["file_name"]
        dst_img  = out / "images" / split / meta["file_name"]
        shutil.copy2(src_img, dst_img)

        img_w, img_h = meta["width"], meta["height"]
        label_lines: list[str] = []

        for ann in anns_by_img.get(img_id, []):
            seg = ann.get("segmentation")
            if seg is None:
                continue

            # 支持 RLE dict 和 polygon list 两种格式
            if isinstance(seg, dict):
                try:
                    mask = rle_to_mask(seg)
                except Exception as e:
                    print(f"  [警告] 跳过 ann {ann['id']}，RLE 解码失败: {e}")
                    skipped += 1
                    continue
                poly = mask_to_polygon(mask, img_w, img_h)
            elif isinstance(seg, list) and seg:
                # polygon 格式：[[x1,y1,x2,y2,...], ...]
                flat = seg[0]  # 取第一个（最大）轮廓
                pts  = np.array(flat, dtype=float).reshape(-1, 2)
                pts[:, 0] /= img_w
                pts[:, 1] /= img_h
                pts = np.clip(pts, 0.0, 1.0)
                poly = pts.flatten().tolist() if len(pts) >= 3 else None
            else:
                continue

            if poly is None or len(poly) < 6:
                skipped += 1
                continue

            cls_idx = cat_to_idx[ann["category_id"]]
            pts_str = " ".join(f"{v:.6f}" for v in poly)
            label_lines.append(f"{cls_idx} {pts_str}")

        stem     = Path(meta["file_name"]).stem
        txt_path = out / "labels" / split / f"{stem}.txt"
        txt_path.write_text("\n".join(label_lines), encoding="utf-8")

    if skipped:
        print(f"[转换] 跳过了 {skipped} 个无效 annotation（mask 太小或解码失败）")

    n_train = len(image_ids) - n_val
    print(f"[转换完成] train={n_train}  val={n_val}  类别={class_names}")
    return {"names": class_names, "output_dir": str(out)}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 生成 dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_yaml(output_dir: str, class_names: list[str]) -> str:
    out  = Path(output_dir).resolve()
    yaml = (
        f"path: {out}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml, encoding="utf-8")
    print(f"[YAML] 已写入 {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 训练
# ─────────────────────────────────────────────────────────────────────────────

def train(yaml_path: str) -> None:
    model = YOLO(BASE_MODEL)
    model.train(
        data         = yaml_path,
        epochs       = EPOCHS,
        imgsz        = IMG_SIZE,
        batch        = BATCH,
        project      = TRAIN_PROJECT,
        name         = TRAIN_NAME,
        exist_ok     = True,
        device       = "0",
        workers      = 4,
        # 早停
        patience     = 20,
        # 学习率
        lr0          = 1e-5,
        weight_decay = 0.001,
        # 数据增强
        hsv_h        = 0.1,
        hsv_s        = 0.1,
        hsv_v        = 0.1,
        degrees      = 10.0,
        flipud       = 0.5,
        fliplr       = 0.5,
        shear        = 0.1,
        mosaic       = 0.0,
        mixup        = 0.0,
    )
    best_pt = Path(TRAIN_PROJECT) / TRAIN_NAME / "weights" / "best.pt"
    print(f"\n[完成] 最佳权重: {best_pt.resolve()}")
    print("使用示例:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO(r'{best_pt}')")
    print(f"  results = model(r'your_video.mp4', stream=True)")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("=" * 60)
    print("Step 1: 转换数据集 COCO → YOLO 分割格式")
    print("=" * 60)
    info = convert_dataset(
        coco_json  = COCO_JSON,
        images_dir = IMAGES_DIR,
        output_dir = OUTPUT_DIR,
        val_ratio  = VAL_RATIO,
        seed       = RANDOM_SEED,
    )

    print("\n" + "=" * 60)
    print("Step 2: 生成 dataset.yaml")
    print("=" * 60)
    yaml_path = write_yaml(info["output_dir"], info["names"])

    print("\n" + "=" * 60)
    print("Step 3: 开始训练")
    print("=" * 60)
    train(yaml_path)
