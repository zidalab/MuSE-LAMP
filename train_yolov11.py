"""
YOLOv11 液滴实例分割训练脚本

流程：
  1. 将 COCO 标注转换为 YOLO 分割格式（已转换则自动跳过）
  2. 生成 dataset.yaml
  3. 启动 YOLO-seg 训练
"""

import json
import multiprocessing
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# 数据集路径
# ─────────────────────────────────────────────────────────────────────────────

COCO_JSON   = r"dataset_seg/_annotations.coco.json"
IMAGES_DIR  = r"dataset_seg/images"
OUTPUT_DIR  = r"dataset_seg_yolo"
VAL_RATIO   = 0.2
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 配置类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseTrainingConfig:
    """所有 YOLO 训练任务的通用参数。"""
    data: str
    epochs: int      = 100
    batch: int       = 16
    imgsz: int       = 640
    workers: int     = min(8, multiprocessing.cpu_count())
    device: str      = '0'
    optimizer: str   = 'auto'
    lr0: float       = 1e-3
    lrf: float       = 0.01
    weight_decay: float = 0.001
    resume: bool     = False
    cache: bool      = False
    patience: int    = 50
    plots: bool      = True
    save_period: int = 20
    amp: bool        = True

    def to_dict(self) -> dict:
        """转为 model.train() 所需的参数字典（排除构造器专用字段）。"""
        exclude = {'model', 'task'}
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k not in exclude}


@dataclass
class SegmentationConfig(BaseTrainingConfig):
    """液滴实例分割配置"""
    task: str      = 'segment'
    model: str     = r'yolo11n-seg.pt'
    project: str   = 'seg_runs'
    name: str      = 'droplet_seg_v2_opt'
    exist_ok: bool = True

    # 掩膜质量
    retina_masks: bool = True
    mask_ratio: int    = 2       # 4→2，掩膜分辨率加倍

    # 学习率
    cos_lr: bool       = True    # 余弦退火，消除后期震荡
    warmup_epochs: int = 3

    # 数据增强
    hsv_h: float       = 0.1
    hsv_s: float       = 0.5    # 适度加强饱和度扰动
    hsv_v: float       = 0.35   # 模拟显微镜光照变化
    degrees: float     = 10.0
    translate: float   = 0.1
    scale: float       = 0.4    # 0.1→0.4，覆盖不同焦距下液滴大小差异
    shear: float       = 0.1
    perspective: float = 0.0
    flipud: float      = 0.5
    fliplr: float      = 0.5
    mosaic: float      = 0.8    # 0→0.8，模拟液滴密疏场景变化
    close_mosaic: int  = 15     # 最后 15 epoch 关闭 mosaic 稳定收敛
    erasing: float     = 0.4    # 随机擦除，模拟液滴遮挡
    copy_paste: float  = 0.3    # 实例复制粘贴，扩充液滴样本多样性
    mixup: float       = 0.0


def get_droplets_seg_config(yaml_path: str) -> SegmentationConfig:
    """返回液滴分割训练配置。"""
    return SegmentationConfig(
        data=yaml_path,
        epochs=200,
        batch=8,
        imgsz=640,
        lr0=0.001,
        weight_decay=0.001,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 数据集转换工具
# ─────────────────────────────────────────────────────────────────────────────

def rle_to_mask(rle: dict) -> np.ndarray:
    """将 COCO RLE 解码为二值掩膜 (H, W)。优先使用 pycocotools。"""
    try:
        from pycocotools import mask as mask_util
        return mask_util.decode(rle).astype(np.uint8)
    except ImportError:
        pass
    if isinstance(rle["counts"], list):
        h, w = rle["size"]
        mask = np.zeros(h * w, dtype=np.uint8)
        pos, fill = 0, 0
        for count in rle["counts"]:
            if fill:
                mask[pos: pos + count] = 1
            pos += count
            fill ^= 1
        return mask.reshape((h, w), order="F")
    raise ValueError("counts 为字符串（compressed RLE），请安装 pycocotools: pip install pycocotools")


def mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int) -> list[float] | None:
    """将二值掩膜转换为归一化 YOLO 多边形点列表 [x1,y1,x2,y2,...]。
    取面积最大的连通域轮廓，approxPolyDP 适当简化以保留边界细节。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
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
    """将 COCO 数据集转为 YOLO 分割格式，返回类别名列表和输出路径信息。"""
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    categories     = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_meta     = {img["id"]: img for img in coco["images"]}
    sorted_cat_ids = sorted(categories.keys())
    cat_to_idx     = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}
    class_names    = [categories[cid] for cid in sorted_cat_ids]

    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

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
        split   = "val" if img_id in val_ids else "train"
        src_img = Path(images_dir) / meta["file_name"]
        dst_img = out / "images" / split / meta["file_name"]
        shutil.copy2(src_img, dst_img)

        img_w, img_h  = meta["width"], meta["height"]
        label_lines: list[str] = []

        for ann in anns_by_img.get(img_id, []):
            seg = ann.get("segmentation")
            if seg is None:
                continue
            if isinstance(seg, dict):
                try:
                    mask = rle_to_mask(seg)
                except Exception as e:
                    print(f"  [警告] 跳过 ann {ann['id']}，RLE 解码失败: {e}")
                    skipped += 1
                    continue
                poly = mask_to_polygon(mask, img_w, img_h)
            elif isinstance(seg, list) and seg:
                flat = seg[0]
                pts  = np.array(flat, dtype=float).reshape(-1, 2)
                pts[:, 0] /= img_w
                pts[:, 1] /= img_h
                pts  = np.clip(pts, 0.0, 1.0)
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
        print(f"[转换] 跳过了 {skipped} 个无效 annotation")

    n_train = len(image_ids) - n_val
    print(f"[转换完成] train={n_train}  val={n_val}  类别={class_names}")
    return {"names": class_names, "output_dir": str(out)}


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
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def train_model(config: SegmentationConfig) -> None:
    warnings.filterwarnings("ignore")
    model = YOLO(config.model, task=config.task)
    model.train(**config.to_dict())
    best_pt = Path(config.project) / config.name / "weights" / "best.pt"
    print(f"\n[完成] 最佳权重: {best_pt.resolve()}")
    print(f"使用示例:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO(r'{best_pt}')")
    print(f"  results = model(r'your_video.mp4', stream=True)")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.freeze_support()

    # Step 1: 转换数据集（已存在则跳过，避免重复 I/O）
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    if yaml_path.exists():
        print(f"[Step 1] 数据集已存在，跳过转换 → {yaml_path}")
    else:
        print("[Step 1] 转换数据集 COCO → YOLO 分割格式")
        info = convert_dataset(COCO_JSON, IMAGES_DIR, OUTPUT_DIR, VAL_RATIO, RANDOM_SEED)
        write_yaml(info["output_dir"], info["names"])

    # Step 2: 训练
    print("\n[Step 2] 开始训练")
    config = get_droplets_seg_config(str(yaml_path))
    train_model(config)


if __name__ == "__main__":
    main()
