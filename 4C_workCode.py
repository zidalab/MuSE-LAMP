"""
Video Tracking and Droplet Analysis Pipeline (Binary Classification Version)
针对单靶标二分类 (0=Negative, 1=Positive)

液滴半径计算参数(需要确认):SCALE_BAR、
                        quantification.py里的area_um2 * 40 * 1e-9中的40指的是通道高度
                        D_CR(代码会根据液滴直径是否大于 D_CR 自动切换计算体积的公式。如果这个阈值设置不准，会导致临界大小的液滴体积计算出现跳变。）
"""

import os
import time
import math
import sys
import subprocess
import traceback
import gc
from itertools import combinations
import cv2
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm
import timm

# -----------------------------------------------------------------------------
# User Configuration (在此处修改路径和参数)
# -----------------------------------------------------------------------------

# 默认测试视频；请从项目根目录运行。可替换为其他视频路径。
VIDEO = [
    r'Sample/SAMPLE.mp4',
]


# 模型路径 (请确认路径是否正确)
DETECTION_MODEL_PATH = r"runs/segment/seg_runs/droplet_seg_v2/weights/best.pt"
# 这里改为你训练好的二分类模型路径
CLASSIFICATION_MODEL_PATH = r"convnextv2b_cls_runs/ConvNeXtV2-B/best.pt"
# 显式启用 BoT-SORT ReID，使用分割网络的原生外观特征。
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_reid.yaml")

# -----------------------------------------------------------------------------
# End of Configuration
# -----------------------------------------------------------------------------

# 尝试导入计算模块
try:
    # 如果失败，尝试从 units 包导入 (兼容旧结构)
    from units.quantification import calculate_volume, get_concentration

    print("Success: Loaded quantification module from units package.")
except ImportError:
    # 如果都失败了，打印红色警告！
    print("\n" + "!" * 50)
    print("ERROR: Could not import 'quantification.py'.")
    print("Calculations will return 0. Please check file structure.")
    print("!" * 50 + "\n")


    # 占位函数 (依然返回0，但你已经看到了警告)
    def calculate_volume(area, scale, d_cr):
        return 0.0


    def get_concentration(pos, neg):
        return 0.0

# 1. 修改类别映射 (二分类)
CLASS_NAME_MAP = {
    '0': 'negative',
    '1': 'positive'
}

# Global constants
SCALE_BAR = 1.72  # um/px
D_CR = 170  # Critical diameter
CROP_SIZE = 128
MAX_SAVE_PER_ID = 5
MIN_CIRCULARITY = 0.8
MIN_MASK_AREA = 1000  # px²，小于此面积的 mask 直接跳过
RUNTIME_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CLASSIFICATION_BATCH_SIZE = 32
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7
CAPTURE_BUFFER_SIZE = 1
FRAME_INTERVAL = 2
MEMORY_CLEANUP_INTERVAL = 300
TRACK_PRUNE_INTERVAL = 100   # 每隔多少帧做一次已完成轨迹清理
TRACK_IDLE_THRESHOLD = 60    # 超过此帧数未出现的已完成轨迹视为已离开视野
GPU_MAX_WORKERS = 1
SAVE_ANNOTATED_VIDEO = True
SAVE_DEBUG_CROPS = True
SUMMARY_EXCEL_FILENAME = "concentration_summary.xlsx"
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

# 四流道实验参数
CHANNEL_COUNT = 4
CHANNEL_FLOW_AXIS = "auto"  # auto / horizontal / vertical
CHANNEL_HOUGH_MODE = "auto"     # auto / wall_pairs / boundaries / centers / equal_split
CHANNEL_LINE_ANGLE_TOL_DEG = 15
CHANNEL_LINE_CLUSTER_DISTANCE_PX = 25
CHANNEL_MIN_LINE_LENGTH_RATIO = 0.15
CHANNEL_BLACKHAT_KERNEL_WIDTH = 31
CHANNEL_BLACKHAT_KERNEL_HEIGHT = 1
CHANNEL_BLACKHAT_THRESHOLD = 25
CHANNEL_BLACKHAT_OPEN_WIDTH = 1
CHANNEL_BLACKHAT_OPEN_HEIGHT = 20
CHANNEL_HOUGH_THRESHOLD = 200
CHANNEL_HOUGH_MAX_LINE_GAP = 30
CHANNEL_MIN_CLUSTER_LINE_COUNT = 2
CHANNEL_MIN_CLUSTER_STRENGTH_RATIO = 0.25
CHANNEL_MAX_WIDTH_RATIO = 1.5
DRAW_CHANNEL_OVERLAY = True
CHANNEL_LAYOUT_BOOTSTRAP_FRAMES = 30  # 在全视频范围内均匀采样，用于跨帧聚合流道边界
CHANNEL_LAYOUT_SAMPLE_START_RATIO = 0.02  # 避开视频最开头可能尚未稳定的画面
CHANNEL_LAYOUT_SAMPLE_END_RATIO = 0.98
CHANNEL_TEMPORAL_CLUSTER_DISTANCE_PX = 30  # 不同帧中的同一条流道壁允许的坐标偏差
CHANNEL_TEMPORAL_MIN_SUPPORT_RATIO = 0.20  # 流道壁至少要在多少比例的有效采样帧中出现
CHANNEL_TEMPORAL_MIN_SUPPORT_FRAMES = 4   # 同时设置绝对帧数下限，避免少量偶然线段入选
CHANNEL_TEMPORAL_MAX_ANGLE_DEVIATION_DEG = 3.0  # 排除主流道方向明显漂移的异常帧
CHANNEL_TEMPORAL_MAX_CANDIDATES = 14  # 几何组合搜索前保留的候选边界上限
CHANNEL_MAX_GAP_RATIO = 2.0  # 相邻流道间距（最宽/最窄）的最大容忍值


def configure_worker_runtime():
    cv2.setNumThreads(1)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def check_model_files():
    missing_paths = []
    for path in [DETECTION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, TRACKER_CONFIG_PATH]:
        if not os.path.exists(path):
            missing_paths.append(path)
    return missing_paths



def get_runtime_device_description():
    if RUNTIME_DEVICE.startswith("cuda") and torch.cuda.is_available():
        try:
            device_index = torch.device(RUNTIME_DEVICE).index or 0
        except Exception:
            device_index = 0

        gpu_name = torch.cuda.get_device_name(device_index)
        try:
            total_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
            return f"{RUNTIME_DEVICE} ({gpu_name}, {total_vram_gb:.2f} GB VRAM)"
        except Exception:
            return f"{RUNTIME_DEVICE} ({gpu_name})"
    return "cpu"


def print_runtime_device(prefix):
    print(f"[{prefix}] Runtime device: {get_runtime_device_description()} | pid={os.getpid()}")


def cleanup_runtime_memory(frame_count=None):
    if MEMORY_CLEANUP_INTERVAL <= 0:
        return

    if frame_count is not None and frame_count > 0 and frame_count % MEMORY_CLEANUP_INTERVAL != 0:
        return

    gc.collect()
    if RUNTIME_DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_yolo_trackers(model):
    """Reset tracker state when reusing one YOLO model across multiple videos."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    vid_path = getattr(predictor, "vid_path", None)
    if isinstance(vid_path, list):
        predictor.vid_path = [None] * len(vid_path)


class ResNetClassifierWrapper:
    """Wrap a torchvision ResNet checkpoint so it can be used like the current classifier."""

    def __init__(self, checkpoint_path, device=None):
        self.device = device or RUNTIME_DEVICE
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = checkpoint.get("class_names", ["0", "1"])
        self.img_size = int(checkpoint.get("img_size", 224))
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])
        self.model = self._build_model(checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _build_model(self, checkpoint):
        from torchvision import models
        import torch.nn as nn

        model_name = checkpoint.get("model_name", "resnet50").lower()
        backbone_builders = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }
        if model_name not in backbone_builders:
            raise ValueError(f"Unsupported classification backbone: {model_name}")

        model = backbone_builders[model_name](weights=None)
        in_features = model.fc.in_features
        dropout = float(checkpoint.get("dropout", 0.0))
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, int(checkpoint.get("num_classes", len(self.class_names)))),
        )
        return model

    def _preprocess_batch(self, images_bgr):
        tensors = []
        for image_bgr in images_bgr:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True) / 255.0
        return (batch - self.mean_tensor.to(dtype=batch.dtype)) / self.std_tensor.to(dtype=batch.dtype)

    def predict(self, image_bgr):
        return self.predict_batch([image_bgr])[0]

    def predict_batch(self, images_bgr, batch_size=CLASSIFICATION_BATCH_SIZE):
        predictions = []
        if not images_bgr:
            return predictions

        with torch.inference_mode():
            for start in range(0, len(images_bgr), batch_size):
                batch_images = images_bgr[start:start + batch_size]
                inputs = self._preprocess_batch(batch_images)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                confidences, class_ids = torch.max(probs, dim=1)
                predictions.extend(
                    (int(class_id.item()), float(confidence.item()))
                    for class_id, confidence in zip(class_ids, confidences)
                )

        return predictions

class ConvNeXtV2ClassifierWrapper:
    """Wrap a timm ConvNeXt V2 checkpoint for inference."""

    def __init__(self, checkpoint_path, device=None):
        self.device = device or RUNTIME_DEVICE
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = checkpoint.get("class_names", ["0", "1"])
        self.img_size = int(checkpoint.get("img_size", 224))
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])
        self.model = self._build_model(checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _build_model(self, checkpoint):
        if timm is None:
            raise ImportError(
                "timm is required for ConvNeXt V2 models. "
                "Please install it: pip install timm"
            )

        model_id = checkpoint.get("model_id", "convnextv2_base.fcmae_ft_in22k_in1k")
        num_classes = int(checkpoint.get("num_classes", len(self.class_names)))
        dropout = float(checkpoint.get("dropout", 0.0))

        print(f"Building timm model: {model_id} (num_classes={num_classes}, dropout={dropout})")
        model = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        return model

    def _preprocess_batch(self, images_bgr):
        tensors = []
        for image_bgr in images_bgr:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True) / 255.0
        return (batch - self.mean_tensor.to(dtype=batch.dtype)) / self.std_tensor.to(dtype=batch.dtype)

    def predict(self, image_bgr):
        return self.predict_batch([image_bgr])[0]

    def predict_batch(self, images_bgr, batch_size=CLASSIFICATION_BATCH_SIZE):
        predictions = []
        if not images_bgr:
            return predictions

        with torch.inference_mode():
            for start in range(0, len(images_bgr), batch_size):
                batch_images = images_bgr[start:start + batch_size]
                inputs = self._preprocess_batch(batch_images)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                confidences, class_ids = torch.max(probs, dim=1)
                predictions.extend(
                    (int(class_id.item()), float(confidence.item()))
                    for class_id, confidence in zip(class_ids, confidences)
                )

        return predictions

def load_classification_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in {".pt", ".pth"}:
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_name = checkpoint.get("model_name", "").lower() if isinstance(checkpoint, dict) else ""
            model_family = checkpoint.get("model_family", "").lower() if isinstance(checkpoint, dict) else ""

            # ---------- 新增：ConvNeXt V2 检测 ----------
            if (
                isinstance(checkpoint, dict)
                and "state_dict" in checkpoint
                and model_family in {"convnextv2"}
                or model_name in {"convnextv2_base", "convnextv2_small", "convnextv2_tiny"}
            ):
                print(f"Loaded ConvNeXt V2 classification checkpoint ({model_name}).")
                return ConvNeXtV2ClassifierWrapper(model_path, device=RUNTIME_DEVICE)
            # ----------------------------------------------

            if (
                isinstance(checkpoint, dict)
                and "state_dict" in checkpoint
                and model_name in {"resnet18", "resnet50"}
            ):
                print(f"Loaded {model_name.upper()} classification checkpoint.")
                return ResNetClassifierWrapper(model_path, device=RUNTIME_DEVICE)
        except Exception:
            pass

    print("Loaded YOLO classification checkpoint.")
    model = YOLO(model_path)
    model.to(RUNTIME_DEVICE)
    return model



def predict_droplet_class(classifier, image_bgr):
    if isinstance(classifier, (ResNetClassifierWrapper, ConvNeXtV2ClassifierWrapper)):
        return classifier.predict(image_bgr)

    with torch.inference_mode():
        class_result = classifier.predict(image_bgr, verbose=False, device=RUNTIME_DEVICE)

    probs = class_result[0].probs
    return int(probs.top1), float(probs.top1conf.item())


def predict_droplet_classes(classifier, images_bgr, batch_size=CLASSIFICATION_BATCH_SIZE):
    if not images_bgr:
        return []

    if isinstance(classifier, (ResNetClassifierWrapper, ConvNeXtV2ClassifierWrapper)):
        return classifier.predict_batch(images_bgr, batch_size=batch_size)

    predictions = []
    for image_bgr in images_bgr:
        predictions.append(predict_droplet_class(classifier, image_bgr))
    return predictions


# -----------------------------------------------------------------------------
# Frame utilities
# -----------------------------------------------------------------------------

def pad_frame(frame, target_h=960, target_w=1280):
    h, w = frame.shape[:2]
    top = max(0, (target_h - h) // 2)
    bottom = max(0, target_h - h - top)
    left = max(0, (target_w - w) // 2)
    right = max(0, target_w - w - left)

    if h <= target_h and w <= target_w:
        return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(frame, (target_w, target_h))


# -----------------------------------------------------------------------------
# 四流道通道布局检测
# -----------------------------------------------------------------------------

def build_channel_edge_map(gray_frame):
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CHANNEL_BLACKHAT_KERNEL_WIDTH, CHANNEL_BLACKHAT_KERNEL_HEIGHT),
    )
    enhanced = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, edges = cv2.threshold(enhanced, CHANNEL_BLACKHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CHANNEL_BLACKHAT_OPEN_WIDTH, CHANNEL_BLACKHAT_OPEN_HEIGHT),
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, open_kernel)
    return edges


def normalize_line_angle(angle_rad):
    while angle_rad < -np.pi / 2:
        angle_rad += np.pi
    while angle_rad >= np.pi / 2:
        angle_rad -= np.pi
    return angle_rad


def angle_delta_deg(angle_a, angle_b):
    delta = normalize_line_angle(angle_a - angle_b)
    return abs(np.degrees(delta))


def get_manual_channel_theta():
    if CHANNEL_FLOW_AXIS == "horizontal":
        return 0.0
    if CHANNEL_FLOW_AXIS == "vertical":
        return np.pi / 2
    return None


def get_oriented_normal(theta):
    nx = -np.sin(theta)
    ny = np.cos(theta)
    if abs(nx) > abs(ny):
        if nx < 0:
            nx, ny = -nx, -ny
    elif ny < 0:
        nx, ny = -nx, -ny
    return np.array([nx, ny], dtype=np.float32)


def point_channel_coord(point, normal):
    return float(point[0] * normal[0] + point[1] * normal[1])


def get_coord_range(width, height, normal):
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    coords = [point_channel_coord(corner, normal) for corner in corners]
    return min(coords), max(coords)


def cluster_line_coords(line_items):
    if not line_items:
        return []

    sorted_items = sorted(line_items, key=lambda item: item["coord"])
    groups = []
    current = [sorted_items[0]]

    for item in sorted_items[1:]:
        last_mean = np.average(
            [g["coord"] for g in current],
            weights=[g["length"] for g in current],
        )
        if abs(item["coord"] - last_mean) <= CHANNEL_LINE_CLUSTER_DISTANCE_PX:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    clusters = []
    for group in groups:
        weights = [item["length"] for item in group]
        coords = [item["coord"] for item in group]
        line_indices = [item.get("line_index", -1) for item in group]
        clusters.append({
            "coord": float(np.average(coords, weights=weights)),
            "coord_min": float(min(coords)),
            "coord_max": float(max(coords)),
            "strength": float(np.sum(weights)),
            "line_count": len(group),
            "line_indices": ",".join(str(i) for i in line_indices),
        })
    return clusters


def filter_line_clusters(clusters, frame_length):
    min_strength = frame_length * CHANNEL_MIN_CLUSTER_STRENGTH_RATIO
    return [
        c for c in clusters
        if c["line_count"] >= CHANNEL_MIN_CLUSTER_LINE_COUNT and c["strength"] >= min_strength
    ]


def choose_top_clusters(clusters, count):
    selected = sorted(clusters, key=lambda c: c["strength"], reverse=True)[:count]
    return sorted(selected, key=lambda c: c["coord"])


def build_lanes_from_wall_pairs(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT * 2:
        return None
    selected = choose_top_clusters(clusters, CHANNEL_COUNT * 2)
    lanes = []
    for i in range(CHANNEL_COUNT):
        left = selected[i * 2]["coord"]
        right = selected[i * 2 + 1]["coord"]
        if right <= left:
            continue
        lanes.append({
            "channel_id": i + 1,
            "min_coord": max(coord_min, left),
            "max_coord": min(coord_max, right),
            "center_coord": (left + right) / 2,
        })
    return lanes if len(lanes) == CHANNEL_COUNT else None


def build_lanes_from_boundaries(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT + 1:
        return None
    selected = choose_top_clusters(clusters, CHANNEL_COUNT + 1)
    lanes = []
    for i in range(CHANNEL_COUNT):
        left = selected[i]["coord"]
        right = selected[i + 1]["coord"]
        if right <= left:
            continue
        lanes.append({
            "channel_id": i + 1,
            "min_coord": max(coord_min, left),
            "max_coord": min(coord_max, right),
            "center_coord": (left + right) / 2,
        })
    return lanes if len(lanes) == CHANNEL_COUNT else None


def build_lanes_from_centers(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT:
        return None
    selected = choose_top_clusters(clusters, CHANNEL_COUNT)
    centers = [c["coord"] for c in selected]
    boundaries = [coord_min]
    for l, r in zip(centers, centers[1:]):
        boundaries.append((l + r) / 2)
    boundaries.append(coord_max)
    lanes = []
    for i, center in enumerate(centers):
        lanes.append({
            "channel_id": i + 1,
            "min_coord": boundaries[i],
            "max_coord": boundaries[i + 1],
            "center_coord": center,
        })
    return lanes


def build_equal_split_lanes(coord_min, coord_max):
    step = (coord_max - coord_min) / CHANNEL_COUNT
    lanes = []
    for i in range(CHANNEL_COUNT):
        left = coord_min + step * i
        right = coord_min + step * (i + 1)
        lanes.append({
            "channel_id": i + 1,
            "min_coord": left,
            "max_coord": right,
            "center_coord": (left + right) / 2,
        })
    return lanes


def validate_lane_widths(lanes):
    if not lanes or len(lanes) < 2:
        return True
    widths = [lane["max_coord"] - lane["min_coord"] for lane in lanes]
    min_w = min(widths)
    if min_w <= 0:
        return False
    return (max(widths) / min_w) <= CHANNEL_MAX_WIDTH_RATIO


def infer_channel_lanes(clusters, coord_min, coord_max):
    mode = CHANNEL_HOUGH_MODE
    lanes = None
    used_mode = mode

    if mode in {"wall_pairs", "auto"}:
        candidate = build_lanes_from_wall_pairs(clusters, coord_min, coord_max)
        if candidate is not None and validate_lane_widths(candidate):
            lanes = candidate
            used_mode = "wall_pairs"
    if lanes is None and mode in {"boundaries", "auto"}:
        candidate = build_lanes_from_boundaries(clusters, coord_min, coord_max)
        if candidate is not None and validate_lane_widths(candidate):
            lanes = candidate
            used_mode = "boundaries"
    if lanes is None and mode in {"centers", "auto"}:
        candidate = build_lanes_from_centers(clusters, coord_min, coord_max)
        if candidate is not None and validate_lane_widths(candidate):
            lanes = candidate
            used_mode = "centers"
    if lanes is None:
        lanes = build_equal_split_lanes(coord_min, coord_max)
        used_mode = "equal_split"

    return lanes, used_mode


def detect_channel_layout(gray_frame):
    height, width = gray_frame.shape[:2]
    edges = build_channel_edge_map(gray_frame)
    min_line_length = int(max(width, height) * CHANNEL_MIN_LINE_LENGTH_RATIO)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=CHANNEL_HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=CHANNEL_HOUGH_MAX_LINE_GAP,
    )

    manual_theta = get_manual_channel_theta()
    line_records = []
    if raw_lines is not None:
        for line in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            angle = normalize_line_angle(np.arctan2(dy, dx))
            line_records.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length, "angle": angle})

    if manual_theta is not None:
        theta = manual_theta
    elif line_records:
        sum_sin = sum(r["length"] * np.sin(2 * r["angle"]) for r in line_records)
        sum_cos = sum(r["length"] * np.cos(2 * r["angle"]) for r in line_records)
        theta = normalize_line_angle(0.5 * np.arctan2(sum_sin, sum_cos))
    else:
        theta = 0.0

    normal = get_oriented_normal(theta)
    coord_min, coord_max = get_coord_range(width, height, normal)
    line_items = []
    for line_index, record in enumerate(line_records):
        angle_delta = angle_delta_deg(record["angle"], theta)
        mid_x = (record["x1"] + record["x2"]) / 2
        mid_y = (record["y1"] + record["y2"]) / 2
        coord = point_channel_coord((mid_x, mid_y), normal)
        record.update({
            "line_index": line_index,
            "angle_deg": float(np.degrees(record["angle"])),
            "angle_delta_deg": float(angle_delta),
            "mid_x": float(mid_x),
            "mid_y": float(mid_y),
            "coord": float(coord),
            "accepted_parallel": angle_delta <= CHANNEL_LINE_ANGLE_TOL_DEG,
        })
        if not record["accepted_parallel"]:
            continue
        line_items.append({"coord": coord, "length": record["length"], "line_index": line_index})

    all_clusters = cluster_line_coords(line_items)
    for cluster_index, cluster in enumerate(all_clusters):
        cluster["cluster_index"] = cluster_index
        cluster["accepted_cluster"] = False

    clusters = filter_line_clusters(all_clusters, max(width, height))
    for filtered_index, cluster in enumerate(clusters):
        cluster["filtered_cluster_index"] = filtered_index
        cluster["accepted_cluster"] = True

    lanes, used_mode = infer_channel_lanes(clusters, coord_min, coord_max)

    return {
        "theta": theta,
        "normal": normal,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "raw_line_count": 0 if raw_lines is None else len(raw_lines),
        "parallel_line_count": len(line_items),
        "all_cluster_count": len(all_clusters),
        "raw_lines": line_records,
        "all_clusters": all_clusters,
        "clusters": clusters,
        "lanes": lanes,
        "mode": used_mode,
    }


def assign_channel_id(point_xy, channel_layout):
    coord = point_channel_coord(point_xy, channel_layout["normal"])
    lanes = channel_layout["lanes"]
    for lane in lanes:
        if lane["min_coord"] <= coord <= lane["max_coord"]:
            return lane["channel_id"], coord
    nearest_lane = min(lanes, key=lambda lane: abs(coord - lane["center_coord"]))
    return nearest_lane["channel_id"], coord


def line_segment_for_coord(coord, normal, width, height):
    nx, ny = float(normal[0]), float(normal[1])
    points = []
    if abs(ny) > 1e-6:
        for x in [0, width - 1]:
            y = (coord - nx * x) / ny
            if 0 <= y <= height - 1:
                points.append((int(round(x)), int(round(y))))
    if abs(nx) > 1e-6:
        for y in [0, height - 1]:
            x = (coord - ny * y) / nx
            if 0 <= x <= width - 1:
                points.append((int(round(x)), int(round(y))))

    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)
    if len(unique_points) < 2:
        return None

    max_pair, max_dist = None, -1
    for idx, pa in enumerate(unique_points):
        for pb in unique_points[idx + 1:]:
            dist = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
            if dist > max_dist:
                max_dist = dist
                max_pair = (pa, pb)
    return max_pair


def draw_channel_overlay(image, channel_layout):
    if not DRAW_CHANNEL_OVERLAY:
        return image

    height, width = image.shape[:2]
    overlay = image.copy()
    colors = [(64, 180, 255), (80, 220, 120), (255, 190, 70), (210, 110, 255)]

    for lane in channel_layout["lanes"]:
        color = colors[(lane["channel_id"] - 1) % len(colors)]
        for coord_key in ["min_coord", "max_coord"]:
            segment = line_segment_for_coord(lane[coord_key], channel_layout["normal"], width, height)
            if segment is not None:
                cv2.line(overlay, segment[0], segment[1], color, 2, cv2.LINE_AA)

        center_segment = line_segment_for_coord(lane["center_coord"], channel_layout["normal"], width, height)
        if center_segment is not None:
            cx = int((center_segment[0][0] + center_segment[1][0]) / 2)
            cy = int((center_segment[0][1] + center_segment[1][1]) / 2)
            cv2.putText(
                overlay, f"C{lane['channel_id']}",
                (max(5, cx - 20), max(25, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"channel mode: {channel_layout['mode']}, lines: {channel_layout['parallel_line_count']}",
        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return overlay


def crop_with_mask(frame_count, results, image, channel_save_dirs, track_id_counter, track_id_info, track_id_pending,
                   track_id_last_seen, wh=CROP_SIZE, label=None, classifier=None, channel_layout=None):
    """
    裁剪、锐化并分类液滴，同时分配流道 ID。
    channel_save_dirs: dict {channel_id: save_dir_path}
    """
    img_height, img_width = image.shape[:2]

    if results[0].masks is None or results[0].boxes.id is None:
        return

    masks = results[0].masks.data.to(torch.uint8).cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    candidates = []

    for mask, box, track_id in zip(masks, boxes, track_ids):
        if track_id is None:
            continue

        track_id_last_seen[track_id] = frame_count

        if track_id in track_id_counter and track_id_counter[track_id] >= MAX_SAVE_PER_ID:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, min(x1, img_width)), max(0, min(y1, img_height))
        x2, y2 = max(0, min(x2, img_width)), max(0, min(y2, img_height))

        if x2 <= x1 or y2 <= y1:
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        channel_id = assign_channel_id((center_x, center_y), channel_layout)[0] if channel_layout is not None else 0

        image_roi = image[y1:y2, x1:x2]
        mask_roi = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
        if image_roi.size == 0 or mask_roi.size == 0:
            continue

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)

        if area < MIN_MASK_AREA:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < MIN_CIRCULARITY:
            continue

        area_um2 = area * SCALE_BAR * SCALE_BAR
        d_eq = 2 * np.sqrt(area_um2 / np.pi)
        if d_eq > 180:
            continue

        final_crop = np.full_like(image_roi, 255)
        final_crop[mask_roi > 0] = image_roi[mask_roi > 0]

        try:
            resized = cv2.resize(final_crop, (wh, wh), interpolation=cv2.INTER_LINEAR)
        except Exception:
            continue

        sharpened = cv2.filter2D(resized, -1, SHARPEN_KERNEL)
        candidates.append({
            "track_id": track_id,
            "area": area,
            "crop": sharpened,
            "channel_id": channel_id,
        })

    if not candidates:
        return

    if not classifier:
        return

    predictions = predict_droplet_classes(
        classifier,
        [item["crop"] for item in candidates],
        batch_size=CLASSIFICATION_BATCH_SIZE
    )

    for item, prediction in zip(candidates, predictions):
        class_id, confidence = prediction
        if confidence <= CLASSIFICATION_CONFIDENCE_THRESHOLD:
            continue

        track_id = item["track_id"]
        if track_id not in track_id_counter:
            track_id_counter[track_id] = 0

        if track_id_counter[track_id] >= MAX_SAVE_PER_ID:
            continue

        pending = track_id_pending[track_id]
        pending['classes'].append(class_id)
        pending['areas'].append(item["area"])
        pending['confidences'].append(confidence)
        pending['channel_ids'].append(item["channel_id"])
        pending['crops'].append((item["crop"], class_id, confidence, item["channel_id"]))
        track_id_counter[track_id] += 1

        if track_id_counter[track_id] == MAX_SAVE_PER_ID:
            track_id_info[track_id]['classes'].extend(pending['classes'])
            track_id_info[track_id]['areas'].extend(pending['areas'])
            track_id_info[track_id]['confidences'].extend(pending['confidences'])
            track_id_info[track_id]['channel_ids'].extend(pending['channel_ids'])
            if SAVE_DEBUG_CROPS:
                for save_idx, (crop_img, cls, conf, ch_id) in enumerate(pending['crops']):
                    ch_dir = channel_save_dirs.get(ch_id, next(iter(channel_save_dirs.values())))
                    save_path = os.path.join(
                        ch_dir,
                        f"{track_id}_{save_idx}_cls{cls}_conf{conf:.3f}.png"
                    )
                    cv2.imwrite(save_path, crop_img)
            # 释放 pending 中的图像数据，避免内存持续累积
            pending['crops'].clear()
            pending['classes'].clear()
            pending['areas'].clear()
            pending['confidences'].clear()
            pending['channel_ids'].clear()


def apply_clahe(frame, clahe=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def calculate_bounds(df):
    if df.empty or 'area' not in df.columns:
        return 0, 0
    Q1 = df['area'].quantile(0.25)
    Q3 = df['area'].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def build_quantification_result(df, root_dir, label):
    """
    2. 修改为单靶标二分类统计逻辑
    """
    save_path = os.path.join(root_dir, f"{label}_consolidated_results.xlsx")

    if df.empty:
        return None

    # 异常面积过滤
    lower, upper = calculate_bounds(df)
    filtered_df = df[(df['area'] >= lower) & (df['area'] <= upper)]

    # 分类面积列表 (0: Negative, 1: Positive)
    class_areas = {
        0: filtered_df[filtered_df['class'] == 0]['area'].tolist(),
        1: filtered_df[filtered_df['class'] == 1]['area'].tolist()
    }

    # 计算体积
    class_volumes = {
        cls: [calculate_volume(area, SCALE_BAR, D_CR) for area in areas]
        for cls, areas in class_areas.items()
    }

    total_droplets = sum(len(v) for v in class_volumes.values())

    if total_droplets == 0:
        return None

    # 计算平均物理参数
    avg_volume = sum(sum(vols) for vols in class_volumes.values()) / total_droplets
    avg_radius = ((3 * avg_volume * 1E9) / (4 * math.pi)) ** (1 / 3)

    # --- 核心修改：直接计算单靶标浓度 ---
    pos_vols = class_volumes[1] # 阳性液滴体积列表
    neg_vols = class_volumes[0] # 阴性液滴体积列表

    concentration = get_concentration(pos_vols, neg_vols)

    results = {
        'file_path': save_path,
        'Concentration(copies/uL)': concentration, # 单一浓度
        'N_positive': len(pos_vols),
        'N_negative': len(neg_vols),
        'N_total': total_droplets,
        'Positive_Rate': len(pos_vols) / total_droplets if total_droplets > 0 else 0,
        'avg_radius_eq(um)': avg_radius
    }

    return results


def build_channel_summary(tracking_df):
    rows = []
    for channel_id in range(1, CHANNEL_COUNT + 1):
        channel_df = tracking_df[tracking_df["channel_id"] == channel_id]
        negative_count = int((channel_df["class"] == 0).sum()) if not channel_df.empty else 0
        positive_count = int((channel_df["class"] == 1).sum()) if not channel_df.empty else 0
        total_count = positive_count + negative_count
        avg_confidence = float(channel_df["avg_confidence"].mean()) if total_count > 0 else np.nan
        avg_area = float(channel_df["area"].mean()) if total_count > 0 else np.nan
        rows.append({
            "channel_id": channel_id,
            "N_positive": positive_count,
            "N_negative": negative_count,
            "N_total": total_count,
            "Positive_Rate": positive_count / total_count if total_count > 0 else np.nan,
            "avg_confidence": avg_confidence,
            "avg_area": avg_area,
        })
    return pd.DataFrame(rows)


def sample_channel_layout_indices(total_frames):
    """在全视频范围内均匀取样，避免只依赖开头少数帧。"""
    if total_frames <= 0 or CHANNEL_LAYOUT_BOOTSTRAP_FRAMES <= 0:
        return []

    sample_count = min(CHANNEL_LAYOUT_BOOTSTRAP_FRAMES, total_frames)
    if sample_count >= total_frames:
        return list(range(total_frames))

    last_frame = total_frames - 1
    start = int(round(last_frame * CHANNEL_LAYOUT_SAMPLE_START_RATIO))
    end = int(round(last_frame * CHANNEL_LAYOUT_SAMPLE_END_RATIO))
    start = max(0, min(start, last_frame))
    end = max(start, min(end, last_frame))
    return sorted(set(np.linspace(start, end, sample_count, dtype=int).tolist()))


def calculate_consensus_channel_theta(layouts):
    """按每帧有效平行线数量加权，计算跨帧一致的轴向角度。"""
    manual_theta = get_manual_channel_theta()
    if manual_theta is not None:
        return manual_theta

    usable = [layout for layout in layouts if layout["parallel_line_count"] > 0]
    if not usable:
        raise RuntimeError("Channel layout detection failed: no parallel Hough lines were found.")

    sum_sin = sum(
        layout["parallel_line_count"] * np.sin(2 * layout["theta"])
        for layout in usable
    )
    sum_cos = sum(
        layout["parallel_line_count"] * np.cos(2 * layout["theta"])
        for layout in usable
    )
    return normalize_line_angle(0.5 * np.arctan2(sum_sin, sum_cos))


def reproject_cluster_to_normal(cluster, layout, normal):
    """将单帧聚类中的原始线段中点重新投影到统一法向量。"""
    line_index_text = str(cluster.get("line_indices", ""))
    line_indices = {
        int(value)
        for value in line_index_text.split(",")
        if value.strip() and value.strip() != "-1"
    }
    records = [
        record for record in layout["raw_lines"]
        if record.get("line_index") in line_indices
    ]
    if not records:
        return None

    weights = [record["length"] for record in records]
    coords = [
        point_channel_coord((record["mid_x"], record["mid_y"]), normal)
        for record in records
    ]
    return {
        "coord": float(np.average(coords, weights=weights)),
        "strength": float(np.sum(weights)),
        "line_count": len(records),
    }


def cluster_temporal_channel_observations(observations, usable_frame_count):
    """合并不同帧中的同一条流道壁，并以逐帧中位坐标作为最终位置。"""
    if not observations:
        return []

    sorted_observations = sorted(observations, key=lambda item: item["coord"])
    groups = []
    current = [sorted_observations[0]]

    for observation in sorted_observations[1:]:
        current_center = float(np.median([item["coord"] for item in current]))
        if abs(observation["coord"] - current_center) <= CHANNEL_TEMPORAL_CLUSTER_DISTANCE_PX:
            current.append(observation)
        else:
            groups.append(current)
            current = [observation]
    groups.append(current)

    temporal_clusters = []
    for group in groups:
        # 同一帧如果出现多个近邻聚类，只保留强度最高的一个，防止重复计票。
        best_by_frame = {}
        for observation in group:
            frame_index = observation["frame_index"]
            previous = best_by_frame.get(frame_index)
            if previous is None or observation["strength"] > previous["strength"]:
                best_by_frame[frame_index] = observation

        frame_observations = list(best_by_frame.values())
        coords = [item["coord"] for item in frame_observations]
        support_count = len(frame_observations)
        temporal_clusters.append({
            "coord": float(np.median(coords)),
            "coord_min": float(min(coords)),
            "coord_max": float(max(coords)),
            "strength": float(sum(item["strength"] for item in frame_observations)),
            "line_count": int(sum(item["line_count"] for item in frame_observations)),
            "support_count": support_count,
            "support_ratio": support_count / usable_frame_count,
            "frame_indices": ",".join(
                str(frame_index) for frame_index in sorted(best_by_frame)
            ),
        })
    return temporal_clusters


def validate_temporal_wall_geometry(wall_clusters, coord_min, coord_max):
    """验证八条壁是否能形成宽度和间距均合理的四个流道。"""
    if len(wall_clusters) != CHANNEL_COUNT * 2:
        return None

    selected = sorted(wall_clusters, key=lambda cluster: cluster["coord"])
    lanes = build_lanes_from_wall_pairs(selected, coord_min, coord_max)
    if lanes is None or not validate_lane_widths(lanes):
        return None

    coords = [cluster["coord"] for cluster in selected]
    gaps = [coords[index + 1] - coords[index] for index in range(1, len(coords) - 1, 2)]
    if not gaps or min(gaps) <= 0:
        return None
    if max(gaps) / min(gaps) > CHANNEL_MAX_GAP_RATIO:
        return None

    widths = [lane["max_coord"] - lane["min_coord"] for lane in lanes]
    width_cv = float(np.std(widths) / np.mean(widths))
    gap_cv = float(np.std(gaps) / np.mean(gaps))
    return lanes, width_cv, gap_cv


def select_temporal_wall_clusters(temporal_clusters, usable_frame_count, coord_min, coord_max):
    """按时间支持度筛选候选，并搜索几何上最合理的八壁组合。"""
    min_support = max(
        CHANNEL_TEMPORAL_MIN_SUPPORT_FRAMES,
        int(math.ceil(usable_frame_count * CHANNEL_TEMPORAL_MIN_SUPPORT_RATIO)),
    )
    supported = [
        cluster for cluster in temporal_clusters
        if cluster["support_count"] >= min_support
    ]
    ranked = sorted(
        supported,
        key=lambda cluster: (cluster["support_count"], cluster["strength"]),
        reverse=True,
    )[:CHANNEL_TEMPORAL_MAX_CANDIDATES]

    best = None
    required_wall_count = CHANNEL_COUNT * 2
    for wall_combination in combinations(ranked, required_wall_count):
        geometry = validate_temporal_wall_geometry(wall_combination, coord_min, coord_max)
        if geometry is None:
            continue
        lanes, width_cv, gap_cv = geometry
        score = (
            sum(cluster["support_count"] for cluster in wall_combination),
            min(cluster["support_count"] for cluster in wall_combination),
            -width_cv,
            -gap_cv,
            sum(cluster["strength"] for cluster in wall_combination),
        )
        if best is None or score > best[0]:
            best = (score, list(wall_combination), lanes)

    if best is None:
        supported_summary = ", ".join(
            f"{cluster['coord']:.1f}px/{cluster['support_count']}f"
            for cluster in sorted(supported, key=lambda cluster: cluster["coord"])
        ) or "none"
        raise RuntimeError(
            "Channel layout detection failed: no reliable eight-wall layout was found. "
            f"sampled_usable_frames={usable_frame_count}, min_support={min_support}, "
            f"supported_candidates=[{supported_summary}]. "
            "The equal_split fallback is disabled to prevent incorrect channel assignment."
        )

    _, selected_clusters, lanes = best
    return sorted(selected_clusters, key=lambda cluster: cluster["coord"]), lanes, min_support


def detect_stable_channel_layout(cap, total_frames, clahe):
    """
    在全视频范围均匀采样，并跨帧聚合边界簇。

    只有在八条流道壁得到足够的时间支持且通过几何校验时才返回布局；
    检测失败会显式报错，不再使用整幅画面四等分作为静默兜底。
    """
    indices = sample_channel_layout_indices(total_frames)
    indexed_layouts = []
    frame_shape = None
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            continue
        gray = apply_clahe(pad_frame(frame), clahe=clahe)
        frame_shape = gray.shape[:2]
        indexed_layouts.append((frame_index, detect_channel_layout(gray)))

    if not indexed_layouts or frame_shape is None:
        raise RuntimeError(
            "Channel layout detection failed: none of the sampled video frames could be read."
        )

    layouts = [layout for _, layout in indexed_layouts]
    theta = calculate_consensus_channel_theta(layouts)
    normal = get_oriented_normal(theta)
    height, width = frame_shape
    coord_min, coord_max = get_coord_range(width, height, normal)

    usable_layouts = [
        (frame_index, layout)
        for frame_index, layout in indexed_layouts
        if layout["parallel_line_count"] > 0
        and angle_delta_deg(layout["theta"], theta) <= CHANNEL_TEMPORAL_MAX_ANGLE_DEVIATION_DEG
    ]
    if not usable_layouts:
        raise RuntimeError(
            "Channel layout detection failed: all sampled frames had unstable channel angles."
        )

    observations = []
    for frame_index, layout in usable_layouts:
        for cluster in layout["clusters"]:
            projected = reproject_cluster_to_normal(cluster, layout, normal)
            if projected is None:
                continue
            projected["frame_index"] = frame_index
            observations.append(projected)

    temporal_clusters = cluster_temporal_channel_observations(
        observations,
        len(usable_layouts),
    )
    selected_clusters, lanes, min_support = select_temporal_wall_clusters(
        temporal_clusters,
        len(usable_layouts),
        coord_min,
        coord_max,
    )

    wall_summary = ", ".join(
        f"{cluster['coord']:.1f}px({cluster['support_count']}f)"
        for cluster in selected_clusters
    )
    print(
        f"[Channel layout] sampled={len(indexed_layouts)}, "
        f"usable={len(usable_layouts)}, min_support={min_support}, "
        f"mode=temporal_wall_pairs, walls=[{wall_summary}]"
    )

    return {
        "theta": theta,
        "normal": normal,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "raw_line_count": int(round(np.median([
            layout["raw_line_count"] for _, layout in usable_layouts
        ]))),
        "parallel_line_count": int(round(np.median([
            layout["parallel_line_count"] for _, layout in usable_layouts
        ]))),
        "all_cluster_count": len(temporal_clusters),
        "raw_lines": [],
        "all_clusters": temporal_clusters,
        "clusters": selected_clusters,
        "lanes": lanes,
        "mode": "temporal_wall_pairs",
        "sampled_frame_count": len(indexed_layouts),
        "usable_frame_count": len(usable_layouts),
        "temporal_min_support": min_support,
    }


def process_video(video_path, output_path, channel_save_dirs, model, classifier, label, progress_position=0):
    track_id_counter = {}
    track_id_info = defaultdict(lambda: {'classes': [], 'areas': [], 'confidences': [], 'channel_ids': []})
    track_id_pending = defaultdict(lambda: {'crops': [], 'areas': [], 'classes': [], 'confidences': [], 'channel_ids': []})
    track_id_last_seen = {}
    reset_yolo_trackers(model)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
    except Exception:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 先验证跨帧八壁布局。失败时立即停止，避免创建错误或空的结果视频。
    try:
        channel_layout = detect_stable_channel_layout(cap, total_frames, clahe)
    except Exception:
        cap.release()
        raise
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = None
    if SAVE_ANNOTATED_VIDEO:
        output_ext = os.path.splitext(output_path)[1].lower()
        if output_ext == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 960))

    # 抽帧逻辑（跳帧）
    frame_count = 0

    pbar = tqdm(
        total=total_frames,
        desc=f"Processing {label}",
        unit="frames",
        ncols=100,
        position=progress_position,
        leave=True
    )
    try:
        while cap.isOpened():
            if FRAME_INTERVAL > 1 and frame_count % FRAME_INTERVAL != 0:
                success = cap.grab()
                pbar.update(1)

                if not success:
                    break

                frame_count += 1
                continue

            success, frame = cap.read()
            pbar.update(1)

            if not success:
                break

            frame = pad_frame(frame)

            clahe_frame = apply_clahe(frame, clahe=clahe)
            clahe_rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2RGB)

            results = model.track(
                clahe_rgb, persist=True, verbose=False, device=RUNTIME_DEVICE,
                retina_masks=True, tracker=TRACKER_CONFIG_PATH,
            )
            if out is not None:
                annotated = results[0].plot(line_width=1)
                annotated = draw_channel_overlay(annotated, channel_layout)
                out.write(annotated)

            crop_with_mask(
                frame_count, results, frame, channel_save_dirs, track_id_counter, track_id_info,
                track_id_pending, track_id_last_seen,
                wh=CROP_SIZE, label=label, classifier=classifier, channel_layout=channel_layout
            )

            del results, clahe_frame, clahe_rgb, frame
            if out is not None:
                del annotated

            frame_count += 1

            if frame_count % TRACK_PRUNE_INTERVAL == 0:
                stale_done = [
                    tid for tid, last in track_id_last_seen.items()
                    if frame_count - last > TRACK_IDLE_THRESHOLD
                    and track_id_counter.get(tid, 0) >= MAX_SAVE_PER_ID
                ]
                for tid in stale_done:
                    track_id_pending.pop(tid, None)
                    track_id_counter.pop(tid, None)
                    del track_id_last_seen[tid]

            cleanup_runtime_memory(frame_count)
    finally:
        pbar.close()
        cap.release()
        if out is not None:
            out.release()
        cleanup_runtime_memory()

    # Flush pending droplets that never reached MAX_SAVE_PER_ID (e.g. fast-moving droplets)
    for track_id, pending in track_id_pending.items():
        if not pending['classes']:
            continue
        if track_id_info[track_id]['classes']:
            continue  # already committed at exactly MAX_SAVE_PER_ID
        track_id_info[track_id]['classes'].extend(pending['classes'])
        track_id_info[track_id]['areas'].extend(pending['areas'])
        track_id_info[track_id]['confidences'].extend(pending['confidences'])
        track_id_info[track_id]['channel_ids'].extend(pending['channel_ids'])
        if SAVE_DEBUG_CROPS:
            for save_idx, (crop_img, cls, conf, ch_id) in enumerate(pending['crops']):
                ch_dir = channel_save_dirs.get(ch_id, next(iter(channel_save_dirs.values())))
                save_path = os.path.join(
                    ch_dir,
                    f"{track_id}_{save_idx}_cls{cls}_conf{conf:.3f}.png"
                )
                cv2.imwrite(save_path, crop_img)

    # --- 3. 汇总逻辑 ---
    tracking_data = []
    for track_id, info in track_id_info.items():
        if not info['classes']:
            continue

        # 简单的多数投票
        class_counts = defaultdict(int)
        for c in info['classes']:
            class_counts[c] += 1

        # 找出出现次数最多的类别
        pos_votes = class_counts[1]
        neg_votes = class_counts[0]
        dominant_class = 1 if pos_votes >= neg_votes else 0

        avg_area = np.mean(info['areas'])
        channel_ids = info.get('channel_ids', [])
        channel_id = max(set(channel_ids), key=channel_ids.count) if channel_ids else 0

        tracking_data.append({
            'track_id': track_id,
            'class': dominant_class,
            'area': avg_area,
            'avg_confidence': np.mean(info['confidences']),
            'channel_id': channel_id,
        })

    return pd.DataFrame(tracking_data)


def resolve_output_paths(video_path):
    """根据视频路径生成统一的结果目录和输出文件路径。"""
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    result_dir = os.path.join(video_dir, f"{video_name}_result")
    channel_dirs = {i: os.path.join(result_dir, f"channel_{i}") for i in range(1, CHANNEL_COUNT + 1)}
    return {
        "video_name": video_name,
        "result_dir": result_dir,
        "output_path": os.path.join(result_dir, f"{video_name}_detected{video_ext}"),
        "channel_dirs": channel_dirs,
        "tracking_path": os.path.join(result_dir, f"{video_name}_tracking.xlsx"),
    }


def process_single_video_task(video_path, detection_model=None, classification_model=None, progress_position=0):
    try:
        configure_worker_runtime()
        print_runtime_device(f"Worker video={video_path}")

        if not os.path.exists(video_path):
            return {"video_path": video_path, "status": "missing", "message": f"Video not found: {video_path}"}

        output_paths = resolve_output_paths(video_path)
        video_name = output_paths["video_name"]
        output_path = output_paths["output_path"]
        channel_dirs = output_paths["channel_dirs"]

        os.makedirs(output_paths["result_dir"], exist_ok=True)
        for ch_dir in channel_dirs.values():
            os.makedirs(ch_dir, exist_ok=True)

        if detection_model is None:
            detection_model = YOLO(DETECTION_MODEL_PATH)
            detection_model.to(RUNTIME_DEVICE)
        if classification_model is None:
            classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

        tracking_df = process_video(
            video_path, output_path, channel_dirs,
            detection_model, classification_model, video_name, progress_position=progress_position
        )

        if tracking_df.empty:
            return {"video_path": video_path, "status": "empty", "message": f"No tracking data for {video_path}"}

        # 保存全视频合并 tracking 表（供参考）
        tracking_df.to_excel(output_paths["tracking_path"], index=False)

        # 按流道独立处理：tracking / 定量 / 浓度
        channel_results = []
        for channel_id in range(1, CHANNEL_COUNT + 1):
            ch_df = tracking_df[tracking_df["channel_id"] == channel_id].copy()
            ch_dir = channel_dirs[channel_id]

            ch_tracking_path = os.path.join(ch_dir, f"{video_name}_channel_{channel_id}_tracking.xlsx")
            ch_df.to_excel(ch_tracking_path, index=False)

            ch_quant = build_quantification_result(ch_df, ch_dir, f"{video_name}_channel_{channel_id}")
            ch_consolidated_path = None
            if ch_quant is not None:
                ch_consolidated_path = os.path.join(ch_dir, f"{video_name}_channel_{channel_id}_consolidated_results.xlsx")
                pd.DataFrame([ch_quant]).to_excel(ch_consolidated_path, index=False)

            channel_results.append({
                "channel_id": channel_id,
                "channel_dir": ch_dir,
                "tracking_path": ch_tracking_path,
                "quant_result": ch_quant,
                "consolidated_path": ch_consolidated_path,
            })

        return {
            "video_path": video_path,
            "status": "ok",
            "result_dir": output_paths["result_dir"],
            "output_path": output_path,
            "tracking_path": output_paths["tracking_path"],
            "channel_results": channel_results,
        }
    except Exception:
        return {
            "video_path": video_path,
            "status": "error",
            "message": f"Error processing {video_path}:\n{traceback.format_exc()}"
        }


def print_task_result(result):
    if result["status"] == "ok":
        print(f"Saved results for {result['video_path']} to: {result['result_dir']}")
        for ch in result.get("channel_results", []):
            qr = ch.get("quant_result") or {}
            conc = qr.get("Concentration(copies/uL)")
            n_pos = qr.get("N_positive", 0)
            n_neg = qr.get("N_negative", 0)
            conc_str = f"{conc:.2f} copies/uL" if isinstance(conc, (int, float)) else "N/A"
            print(f"  Channel {ch['channel_id']}: {conc_str}, N_pos={n_pos}, N_neg={n_neg}")
    else:
        print(result["message"])





def get_summary_output_path(video_paths):
    """Return the shared output path for the final concentration summary."""
    video_dirs = [os.path.dirname(os.path.abspath(path)) for path in video_paths]
    try:
        common_dir = os.path.commonpath(video_dirs) if video_dirs else os.getcwd()
    except ValueError:
        common_dir = os.getcwd()
    os.makedirs(common_dir, exist_ok=True)
    return os.path.join(common_dir, SUMMARY_EXCEL_FILENAME)


def export_concentration_summary(task_results, output_path):
    """Export one final Excel sheet with one row per channel per video."""
    summary_rows = []

    for result in task_results:
        video_path = result.get("video_path", "")
        video_name = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
        sample_group = os.path.basename(os.path.dirname(video_path)) if video_path else ""
        status = result.get("status", "")
        message = result.get("message", "")

        channel_results = result.get("channel_results", [])
        if not channel_results:
            # 失败或空结果：保留一行占位
            summary_rows.append({
                "sample_group": sample_group,
                "video_name": video_name,
                "video_path": video_path,
                "channel_id": np.nan,
                "status": status,
                "Concentration(copies/uL)": np.nan,
                "N_positive": np.nan,
                "N_negative": np.nan,
                "N_total": np.nan,
                "Positive_Rate": np.nan,
                "avg_radius_eq(um)": np.nan,
                "consolidated_path": "",
                "message": message,
            })
            continue

        for ch in channel_results:
            qr = ch.get("quant_result") or {}
            summary_rows.append({
                "sample_group": sample_group,
                "video_name": video_name,
                "video_path": video_path,
                "channel_id": ch["channel_id"],
                "status": status,
                "Concentration(copies/uL)": qr.get("Concentration(copies/uL)", np.nan),
                "N_positive": qr.get("N_positive", np.nan),
                "N_negative": qr.get("N_negative", np.nan),
                "N_total": qr.get("N_total", np.nan),
                "Positive_Rate": qr.get("Positive_Rate", np.nan),
                "avg_radius_eq(um)": qr.get("avg_radius_eq(um)", np.nan),
                "consolidated_path": ch.get("consolidated_path") or "",
                "message": message,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(output_path, index=False)
    return output_path


def main():
    start_time = time.time()

    print(f"{'='*50}")
    print("Initializing Droplet Analysis Pipeline (Binary Mode)")
    print(f"{'='*50}")
    print_runtime_device("Main process")

    missing_paths = check_model_files()
    if missing_paths:
        print("Error loading models. Check paths.")
        for path in missing_paths:
            print(f"Missing model file: {path}")
        raise SystemExit(1)

    if not VIDEO:
        print("VIDEO list is empty, nothing to process.")
        return

    print("Loading models once for the whole batch...")
    detection_model = YOLO(DETECTION_MODEL_PATH)
    detection_model.to(RUNTIME_DEVICE)
    classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

    task_results = []
    for idx, video_path in enumerate(VIDEO):
        result = process_single_video_task(
            video_path,
            detection_model=detection_model,
            classification_model=classification_model,
            progress_position=idx
        )
        task_results.append(result)
        print_task_result(result)

    summary_path = get_summary_output_path(VIDEO)
    export_concentration_summary(task_results, summary_path)
    print(f"Saved final concentration summary to:\n{summary_path}")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
