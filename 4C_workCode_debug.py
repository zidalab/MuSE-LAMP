"""
Random-frame debug runner.

This script is independent from 4C_workCode.py. Edit the VIDEO list and model
paths below, then run this file to test only a few random frames from each video.
"""

import gc
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# 在这里修改你要测试的视频路径。
VIDEO = [
    r'E:/VsCode Project/DeepLamp_4channel/data/20260826 PG2/16BK16CMV17EBV17BK.mp4',
]

# 在这里修改模型路径。
DETECTION_MODEL_PATH = r"runs/segment/seg_runs/droplet_seg_v2/weights/best.pt"
CLASSIFICATION_MODEL_PATH = r"convnextv2b_cls_runs/ConvNeXtV2-B/best.pt"

# 随机抽帧测试参数。
DEBUG_VIDEO_LIMIT = None       # None=测试 VIDEO 中所有视频；1=只测试第一个视频。
DEBUG_FRAMES_PER_VIDEO = 8     # 每个视频随机抽几帧。
DEBUG_RANDOM_SEED = 1   # 固定随机种子，方便复现实验。
DEBUG_OUTPUT_DIRNAME = "debug_random_frames"

# 输出开关。
SAVE_DEBUG_ANNOTATED_FRAMES = True
SAVE_DEBUG_CROPS = True
SAVE_DEBUG_EDGE_FRAMES = True
SAVE_DEBUG_HOUGH_LINES = True

# 四流道实验参数。
CHANNEL_COUNT = 4
CHANNEL_FLOW_AXIS = "auto" # auto / horizontal / vertical
CHANNEL_HOUGH_MODE = "auto"    # auto / wall_pairs / boundaries / centers / equal_split
CHANNEL_LINE_ANGLE_TOL_DEG = 15  # Hough线段与主流道方向的角度容差，单位是度，越大，越容易接受倾斜的线段，但可能引入更多噪声线段。根据视频质量和分辨率调整。
CHANNEL_LINE_CLUSTER_DISTANCE_PX = 25  # Hough线段聚类时，线段坐标相差多少像素以内，才有可能属于同一条宏观流道边界。宏观看起来糊成一团时应增大。
CHANNEL_MIN_LINE_LENGTH_RATIO = 0.15  # 流道边界线段的最短长度占画面较长边的比例，越高，检测到的线段越长越明显，但可能漏掉一些短线段。根据视频质量和分辨率调整。
CHANNEL_BLACKHAT_KERNEL_WIDTH = 31   # Black-hat横向核宽度，越大越能增强竖向暗线，但也可能合并邻近结构。
CHANNEL_BLACKHAT_KERNEL_HEIGHT = 1   # Black-hat纵向核高度，竖向流道建议较小，避免过度增强液滴边缘。
CHANNEL_BLACKHAT_THRESHOLD = 25      # Black-hat二值化阈值，越低边缘越多，越高边缘越干净。
CHANNEL_BLACKHAT_OPEN_WIDTH = 1      # 二值图开运算核宽度，用于去掉小噪点。
CHANNEL_BLACKHAT_OPEN_HEIGHT = 20    # 二值图开运算核高度，用于保留竖向连续结构。
CHANNEL_HOUGH_THRESHOLD = 200  # HoughLinesP 阈值，越高，只有更明显的线段会被检测到，检测结果更干净，但可能漏掉一些线段。根据视频质量和分辨率调整。
CHANNEL_HOUGH_MAX_LINE_GAP = 30  # HoughLinesP 连接线段的最大间距，越大，越容易将断续的线段连接成完整的流道边界，但可能错误连接不相关的线段。根据视频质量和分辨率调整。
CHANNEL_MIN_CLUSTER_LINE_COUNT = 2  # 一个流道边界的线段聚类中，至少要有多少条线段，才有可能是真实的流道边界。根据视频质量和分辨率调整。
CHANNEL_MIN_CLUSTER_STRENGTH_RATIO = 0.25  # 一个流道边界的线段聚类的总强度（线段长度之和）至少要占画面较长边的多少比例，才有可能是真实的流道边界。根据视频质量和分辨率调整。
CHANNEL_MAX_WIDTH_RATIO = 1.5  # 各流道宽度之比（最宽/最窄）的最大容忍值。实验中各流道宽度相近，超过此值说明当前检测模式划分有误，将拒绝该结果并降级到下一种模式。
DRAW_CHANNEL_OVERLAY = True


# -----------------------------------------------------------------------------
# Runtime/model configuration
# -----------------------------------------------------------------------------

CLASS_NAME_MAP = {
    "0": "negative",
    "1": "positive",
}

SCALE_BAR = 1.72
CROP_SIZE = 128
MIN_CIRCULARITY = 0.75
RUNTIME_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CLASSIFICATION_BATCH_SIZE = 32
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7
MEMORY_CLEANUP_INTERVAL = 0
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


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
        model_id = checkpoint.get("model_id", "convnextv2_base.fcmae_ft_in22k_in1k")
        num_classes = int(checkpoint.get("num_classes", len(self.class_names)))
        dropout = float(checkpoint.get("dropout", 0.0))

        print(f"Building timm model: {model_id} (num_classes={num_classes}, dropout={dropout})")
        return timm.create_model(
            model_id,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def _preprocess_batch(self, images_bgr):
        tensors = []
        for image_bgr in images_bgr:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True) / 255.0
        return (batch - self.mean_tensor.to(dtype=batch.dtype)) / self.std_tensor.to(dtype=batch.dtype)

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


def configure_worker_runtime():
    cv2.setNumThreads(1)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


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


def cleanup_runtime_memory():
    gc.collect()
    if RUNTIME_DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_model_files():
    missing_paths = [
        path for path in [DETECTION_MODEL_PATH, CLASSIFICATION_MODEL_PATH]
        if not os.path.exists(path)
    ]
    if missing_paths:
        print("Error loading models. Check paths.")
        for path in missing_paths:
            print(f"Missing model file: {path}")
        raise SystemExit(1)


def load_classification_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in {".pt", ".pth"}:
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_name = checkpoint.get("model_name", "").lower() if isinstance(checkpoint, dict) else ""
            model_family = checkpoint.get("model_family", "").lower() if isinstance(checkpoint, dict) else ""

            if (
                isinstance(checkpoint, dict)
                and "state_dict" in checkpoint
                and (model_family == "convnextv2" or model_name.startswith("convnextv2"))
            ):
                print(f"Loaded ConvNeXt V2 classification checkpoint ({model_name}).")
                return ConvNeXtV2ClassifierWrapper(model_path, device=RUNTIME_DEVICE)
        except Exception:
            pass

    print("Loaded YOLO classification checkpoint.")
    model = YOLO(model_path)
    model.to(RUNTIME_DEVICE)
    return model


def predict_droplet_classes(classifier, images_bgr, batch_size=CLASSIFICATION_BATCH_SIZE):
    if not images_bgr:
        return []

    if isinstance(classifier, ConvNeXtV2ClassifierWrapper):
        return classifier.predict_batch(images_bgr, batch_size=batch_size)

    predictions = []
    for image_bgr in images_bgr:
        with torch.inference_mode():
            class_result = classifier.predict(image_bgr, verbose=False, device=RUNTIME_DEVICE)
        probs = class_result[0].probs
        predictions.append((int(probs.top1), float(probs.top1conf.item())))

    return predictions


def apply_clahe(frame, clahe=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resolve_output_paths(video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = os.path.join(video_dir, f"{video_name}_result")
    return {
        "video_name": video_name,
        "result_dir": result_dir,
    }


def pad_frame(frame, target_h=960, target_w=1280):
    h, w = frame.shape[:2]
    top = max(0, (target_h - h) // 2)
    bottom = max(0, target_h - h - top)
    left = max(0, (target_w - w) // 2)
    right = max(0, target_w - w - left)

    if h <= target_h and w <= target_w:
        return cv2.copyMakeBorder(
            frame,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    return cv2.resize(frame, (target_w, target_h))


def sample_frame_indices(total_frames, frames_per_video, rng):
    if total_frames <= 0:
        return []

    sample_count = min(frames_per_video, total_frames)
    return sorted(rng.sample(range(total_frames), sample_count))


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


def get_edge_method_description():
    return (
        "blackhat "
        f"k={CHANNEL_BLACKHAT_KERNEL_WIDTH}x{CHANNEL_BLACKHAT_KERNEL_HEIGHT}, "
        f"thr={CHANNEL_BLACKHAT_THRESHOLD}, "
        f"open={CHANNEL_BLACKHAT_OPEN_WIDTH}x{CHANNEL_BLACKHAT_OPEN_HEIGHT}"
    )


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
    corners = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ]
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
            [group_item["coord"] for group_item in current],
            weights=[group_item["length"] for group_item in current],
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
        clusters.append(
            {
                "coord": float(np.average(coords, weights=weights)),
                "coord_min": float(min(coords)),
                "coord_max": float(max(coords)),
                "strength": float(np.sum(weights)),
                "line_count": len(group),
                "line_indices": ",".join(str(index) for index in line_indices),
            }
        )

    return clusters


def filter_line_clusters(clusters, frame_length):
    min_strength = frame_length * CHANNEL_MIN_CLUSTER_STRENGTH_RATIO
    return [
        cluster for cluster in clusters
        if (
            cluster["line_count"] >= CHANNEL_MIN_CLUSTER_LINE_COUNT
            and cluster["strength"] >= min_strength
        )
    ]


def choose_top_clusters(clusters, count):
    selected = sorted(clusters, key=lambda item: item["strength"], reverse=True)[:count]
    return sorted(selected, key=lambda item: item["coord"])


def build_lanes_from_wall_pairs(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT * 2:
        return None

    selected = choose_top_clusters(clusters, CHANNEL_COUNT * 2)
    lanes = []
    for channel_idx in range(CHANNEL_COUNT):
        left = selected[channel_idx * 2]["coord"]
        right = selected[channel_idx * 2 + 1]["coord"]
        if right <= left:
            continue
        lanes.append(
            {
                "channel_id": channel_idx + 1,
                "min_coord": max(coord_min, left),
                "max_coord": min(coord_max, right),
                "center_coord": (left + right) / 2,
            }
        )

    return lanes if len(lanes) == CHANNEL_COUNT else None


def build_lanes_from_boundaries(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT + 1:
        return None

    selected = choose_top_clusters(clusters, CHANNEL_COUNT + 1)
    lanes = []
    for channel_idx in range(CHANNEL_COUNT):
        left = selected[channel_idx]["coord"]
        right = selected[channel_idx + 1]["coord"]
        if right <= left:
            continue
        lanes.append(
            {
                "channel_id": channel_idx + 1,
                "min_coord": max(coord_min, left),
                "max_coord": min(coord_max, right),
                "center_coord": (left + right) / 2,
            }
        )

    return lanes if len(lanes) == CHANNEL_COUNT else None


def build_lanes_from_centers(clusters, coord_min, coord_max):
    if len(clusters) < CHANNEL_COUNT:
        return None

    selected = choose_top_clusters(clusters, CHANNEL_COUNT)
    centers = [item["coord"] for item in selected]
    boundaries = [coord_min]
    for left, right in zip(centers, centers[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(coord_max)

    lanes = []
    for channel_idx, center in enumerate(centers):
        lanes.append(
            {
                "channel_id": channel_idx + 1,
                "min_coord": boundaries[channel_idx],
                "max_coord": boundaries[channel_idx + 1],
                "center_coord": center,
            }
        )
    return lanes


def build_equal_split_lanes(coord_min, coord_max):
    step = (coord_max - coord_min) / CHANNEL_COUNT
    lanes = []
    for channel_idx in range(CHANNEL_COUNT):
        left = coord_min + step * channel_idx
        right = coord_min + step * (channel_idx + 1)
        lanes.append(
            {
                "channel_id": channel_idx + 1,
                "min_coord": left,
                "max_coord": right,
                "center_coord": (left + right) / 2,
            }
        )
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
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            angle = normalize_line_angle(np.arctan2(dy, dx))
            line_records.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "length": length,
                    "angle": angle,
                }
            )

    if manual_theta is not None:
        theta = manual_theta
    elif line_records:
        sum_sin = sum(record["length"] * np.sin(2 * record["angle"]) for record in line_records)
        sum_cos = sum(record["length"] * np.cos(2 * record["angle"]) for record in line_records)
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
        record["line_index"] = line_index
        record["angle_deg"] = float(np.degrees(record["angle"]))
        record["angle_delta_deg"] = float(angle_delta)
        record["mid_x"] = float(mid_x)
        record["mid_y"] = float(mid_y)
        record["coord"] = float(coord)
        record["accepted_parallel"] = angle_delta <= CHANNEL_LINE_ANGLE_TOL_DEG

        if not record["accepted_parallel"]:
            continue

        line_items.append(
            {
                "coord": coord,
                "length": record["length"],
                "line_index": record["line_index"],
            }
        )

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
        "edges": edges,
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
    for point in points:
        if point not in unique_points:
            unique_points.append(point)

    if len(unique_points) < 2:
        return None

    max_pair = None
    max_dist = -1
    for idx, point_a in enumerate(unique_points):
        for point_b in unique_points[idx + 1:]:
            dist = (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
            if dist > max_dist:
                max_dist = dist
                max_pair = (point_a, point_b)

    return max_pair


def draw_channel_overlay(image, channel_layout):
    if not DRAW_CHANNEL_OVERLAY:
        return image

    height, width = image.shape[:2]
    overlay = image.copy()
    colors = [
        (64, 180, 255),
        (80, 220, 120),
        (255, 190, 70),
        (210, 110, 255),
    ]

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
                overlay,
                f"C{lane['channel_id']}",
                (max(5, cx - 20), max(25, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"channel mode: {channel_layout['mode']}, lines: {channel_layout['parallel_line_count']}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def draw_detection_area_labels(image, frame_rows):
    for row in frame_rows:
        text = f"{int(row['area'])}px"
        x = int(row["center_x"])
        y1 = int(row["y1"])
        cv2.putText(
            image,
            text,
            (max(0, x - 20), max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (max(0, x - 20), max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def save_edge_debug_frame(edge_dir, video_name, frame_index, clahe_frame, edges):
    if not SAVE_DEBUG_EDGE_FRAMES:
        return

    clahe_bgr = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    separator = np.full((clahe_frame.shape[0], 8, 3), 255, dtype=np.uint8)
    preview = np.hstack([clahe_bgr, separator, edges_bgr])

    cv2.putText(
        preview,
        "CLAHE",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        get_edge_method_description(),
        (clahe_frame.shape[1] + 28, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    preview_path = os.path.join(edge_dir, f"{video_name}_f{frame_index:06d}_blackhat_compare.jpg")
    edge_path = os.path.join(edge_dir, f"{video_name}_f{frame_index:06d}_blackhat_only.png")
    cv2.imwrite(preview_path, preview)
    cv2.imwrite(edge_path, edges)


def save_hough_lines_debug_frame(hough_dir, video_name, frame_index, clahe_frame, channel_layout):
    if not SAVE_DEBUG_HOUGH_LINES:
        return

    overlay = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)
    for record in channel_layout["raw_lines"]:
        pt1 = (int(round(record["x1"])), int(round(record["y1"])))
        pt2 = (int(round(record["x2"])), int(round(record["y2"])))
        if record["accepted_parallel"]:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (0, 180, 255)
            thickness = 1
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    cv2.putText(
        overlay,
        (
            f"Hough raw={channel_layout['raw_line_count']}, "
            f"accepted={channel_layout['parallel_line_count']}, "
            f"theta={np.degrees(channel_layout['theta']):.1f}deg"
        ),
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "green=angle accepted, orange=raw only",
        (18, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    line_path = os.path.join(hough_dir, f"{video_name}_f{frame_index:06d}_hough_lines.jpg")
    cv2.imwrite(line_path, overlay)

    cluster_overlay = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)
    height, width = clahe_frame.shape[:2]
    for cluster in channel_layout["all_clusters"]:
        segment = line_segment_for_coord(cluster["coord"], channel_layout["normal"], width, height)
        if segment is None:
            continue

        if cluster["accepted_cluster"]:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 180, 255)
            thickness = 1

        cv2.line(cluster_overlay, segment[0], segment[1], color, thickness, cv2.LINE_AA)
        label_x = int((segment[0][0] + segment[1][0]) / 2)
        label_y = int((segment[0][1] + segment[1][1]) / 2)
        cv2.putText(
            cluster_overlay,
            f"K{cluster['cluster_index']} n={cluster['line_count']}",
            (max(0, label_x - 36), max(22, label_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        cluster_overlay,
        (
            f"Hough clusters all={channel_layout['all_cluster_count']}, "
            f"accepted={len(channel_layout['clusters'])}, "
            f"merge_px={CHANNEL_LINE_CLUSTER_DISTANCE_PX}"
        ),
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        cluster_overlay,
        "green=accepted macro cluster, orange=filtered cluster",
        (18, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cluster_path = os.path.join(hough_dir, f"{video_name}_f{frame_index:06d}_hough_clusters.jpg")
    cv2.imwrite(cluster_path, cluster_overlay)


def build_channel_summary(result_df):
    rows = []
    for channel_id in range(1, CHANNEL_COUNT + 1):
        channel_df = result_df[result_df["channel_id"] == channel_id]
        negative_count = int((channel_df["class"] == 0).sum()) if not channel_df.empty else 0
        positive_count = int((channel_df["class"] == 1).sum()) if not channel_df.empty else 0
        total_count = positive_count + negative_count
        avg_confidence = float(channel_df["confidence"].mean()) if total_count > 0 else np.nan
        avg_area = float(channel_df["area"].mean()) if total_count > 0 else np.nan

        rows.append(
            {
                "channel_id": channel_id,
                "N_positive": positive_count,
                "N_negative": negative_count,
                "N_total": total_count,
                "Positive_Rate": positive_count / total_count if total_count > 0 else np.nan,
                "avg_confidence": avg_confidence,
                "avg_area": avg_area,
            }
        )

    return pd.DataFrame(rows)


def save_crop(crop_dir, video_name, frame_index, det_index, channel_id, class_id, confidence, crop):
    if not SAVE_DEBUG_CROPS:
        return

    crop_path = os.path.join(
        crop_dir,
        f"{video_name}_f{frame_index:06d}_c{channel_id}_det{det_index:03d}_cls{class_id}_conf{confidence:.3f}.png",
    )
    cv2.imwrite(crop_path, crop)


def crop_and_classify_frame(frame, results, classifier, crop_dir, video_name, frame_index, channel_layout):
    if results[0].masks is None or results[0].boxes is None:
        return []

    img_height, img_width = frame.shape[:2]
    masks = results[0].masks.data.to(torch.uint8).cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    candidates = []

    for det_index, (mask, box) in enumerate(zip(masks, boxes)):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, min(x1, img_width)), max(0, min(y1, img_height))
        x2, y2 = max(0, min(x2, img_width)), max(0, min(y2, img_height))

        if x2 <= x1 or y2 <= y1:
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        channel_id, channel_coord = assign_channel_id((center_x, center_y), channel_layout)

        image_roi = frame[y1:y2, x1:x2]
        mask_roi = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
        if image_roi.size == 0 or mask_roi.size == 0:
            continue

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
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
            resized = cv2.resize(final_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        except Exception:
            continue

        sharpened = cv2.filter2D(resized, -1, SHARPEN_KERNEL)
        candidates.append(
            {
                "det_index": det_index,
                "channel_id": channel_id,
                "channel_coord": channel_coord,
                "center_x": center_x,
                "center_y": center_y,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area,
                "circularity": circularity,
                "crop": sharpened,
            }
        )

    if not candidates:
        return []

    predictions = predict_droplet_classes(
        classifier,
        [item["crop"] for item in candidates],
        batch_size=CLASSIFICATION_BATCH_SIZE,
    )

    rows = []
    for item, (class_id, confidence) in zip(candidates, predictions):
        if confidence <= CLASSIFICATION_CONFIDENCE_THRESHOLD:
            continue

        save_crop(
            crop_dir,
            video_name,
            frame_index,
            item["det_index"],
            item["channel_id"],
            class_id,
            confidence,
            item["crop"],
        )
        rows.append(
            {
                "frame_index": frame_index,
                "det_index": item["det_index"],
                "channel_id": item["channel_id"],
                "channel_coord": float(item["channel_coord"]),
                "center_x": float(item["center_x"]),
                "center_y": float(item["center_y"]),
                "x1": int(item["x1"]),
                "y1": int(item["y1"]),
                "x2": int(item["x2"]),
                "y2": int(item["y2"]),
                "class": int(class_id),
                "class_name": CLASS_NAME_MAP.get(str(class_id), str(class_id)),
                "confidence": float(confidence),
                "area": float(item["area"]),
                "circularity": float(item["circularity"]),
            }
        )

    return rows


def process_random_frames(video_path, detection_model, classification_model, rng):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return pd.DataFrame()

    output_paths = resolve_output_paths(video_path)
    video_name = output_paths["video_name"]
    debug_dir = os.path.join(output_paths["result_dir"], DEBUG_OUTPUT_DIRNAME)
    frame_dir = os.path.join(debug_dir, "annotated_frames")
    crop_dir = os.path.join(debug_dir, "crops")
    edge_dir = os.path.join(debug_dir, "edge_frames")
    hough_dir = os.path.join(debug_dir, "hough_lines")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(hough_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = sample_frame_indices(total_frames, DEBUG_FRAMES_PER_VIDEO, rng)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rows = []
    timing_rows = []
    layout_rows = []
    hough_line_rows = []
    hough_cluster_rows = []

    print(f"\nVideo: {video_path}")
    print(f"Total frames: {total_frames}, fps: {fps:.2f}")
    print(f"Sampled frames: {frame_indices}")

    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                print(f"  frame {frame_index}: read failed")
                continue

            frame = pad_frame(frame)
            start = time.perf_counter()

            clahe_frame = apply_clahe(frame, clahe=clahe)
            clahe_rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2RGB)
            channel_layout = detect_channel_layout(clahe_frame)
            save_edge_debug_frame(
                edge_dir,
                video_name,
                frame_index,
                clahe_frame,
                channel_layout["edges"],
            )
            save_hough_lines_debug_frame(
                hough_dir,
                video_name,
                frame_index,
                clahe_frame,
                channel_layout,
            )

            with torch.inference_mode():
                results = detection_model.predict(
                    clahe_rgb,
                    verbose=False,
                    device=RUNTIME_DEVICE,
                    retina_masks=True,
                )

            frame_rows = crop_and_classify_frame(
                frame,
                results,
                classification_model,
                crop_dir,
                video_name,
                frame_index,
                channel_layout,
            )

            elapsed = time.perf_counter() - start
            timing_rows.append(
                {
                    "frame_index": frame_index,
                    "elapsed_s": elapsed,
                    "detections_after_filter": len(frame_rows),
                    "channel_mode": channel_layout["mode"],
                    "hough_raw_lines": channel_layout["raw_line_count"],
                    "hough_parallel_lines": channel_layout["parallel_line_count"],
                    "hough_all_clusters": channel_layout["all_cluster_count"],
                    "hough_clusters": len(channel_layout["clusters"]),
                }
            )
            for lane in channel_layout["lanes"]:
                layout_rows.append(
                    {
                        "frame_index": frame_index,
                        "channel_id": lane["channel_id"],
                        "min_coord": lane["min_coord"],
                        "max_coord": lane["max_coord"],
                        "center_coord": lane["center_coord"],
                        "mode": channel_layout["mode"],
                        "theta_deg": float(np.degrees(channel_layout["theta"])),
                        "raw_line_count": channel_layout["raw_line_count"],
                        "parallel_line_count": channel_layout["parallel_line_count"],
                        "all_cluster_count": channel_layout["all_cluster_count"],
                        "cluster_count": len(channel_layout["clusters"]),
                    }
                )

            for record in channel_layout["raw_lines"]:
                hough_line_rows.append(
                    {
                        "frame_index": frame_index,
                        "line_index": record["line_index"],
                        "x1": record["x1"],
                        "y1": record["y1"],
                        "x2": record["x2"],
                        "y2": record["y2"],
                        "mid_x": record["mid_x"],
                        "mid_y": record["mid_y"],
                        "length": record["length"],
                        "angle_deg": record["angle_deg"],
                        "angle_delta_deg": record["angle_delta_deg"],
                        "coord": record["coord"],
                        "accepted_parallel": record["accepted_parallel"],
                        "channel_theta_deg": float(np.degrees(channel_layout["theta"])),
                        "channel_mode": channel_layout["mode"],
                    }
                )

            for cluster in channel_layout["all_clusters"]:
                hough_cluster_rows.append(
                    {
                        "frame_index": frame_index,
                        "cluster_index": cluster["cluster_index"],
                        "filtered_cluster_index": cluster.get("filtered_cluster_index", np.nan),
                        "coord": cluster["coord"],
                        "coord_min": cluster["coord_min"],
                        "coord_max": cluster["coord_max"],
                        "coord_span": cluster["coord_max"] - cluster["coord_min"],
                        "line_count": cluster["line_count"],
                        "strength": cluster["strength"],
                        "accepted_cluster": cluster["accepted_cluster"],
                        "line_indices": cluster["line_indices"],
                        "cluster_distance_px": CHANNEL_LINE_CLUSTER_DISTANCE_PX,
                        "min_cluster_line_count": CHANNEL_MIN_CLUSTER_LINE_COUNT,
                        "min_cluster_strength_ratio": CHANNEL_MIN_CLUSTER_STRENGTH_RATIO,
                    }
                )

            for row in frame_rows:
                row["video_path"] = video_path
                rows.append(row)

            if SAVE_DEBUG_ANNOTATED_FRAMES:
                annotated = results[0].plot(line_width=1)
                annotated = draw_channel_overlay(annotated, channel_layout)
                annotated = draw_detection_area_labels(annotated, frame_rows)
                annotated_path = os.path.join(frame_dir, f"{video_name}_f{frame_index:06d}.jpg")
                cv2.imwrite(annotated_path, annotated)

            print(
                f"  frame {frame_index:>6}: {elapsed:.3f}s, "
                f"{len(frame_rows)} droplets after filters, "
                f"channel={channel_layout['mode']}, lines={channel_layout['parallel_line_count']}"
            )

            del results, clahe_frame, clahe_rgb, frame, channel_layout
            cleanup_runtime_memory()
    finally:
        cap.release()

    timing_df = pd.DataFrame(timing_rows)
    result_df = pd.DataFrame(rows)
    layout_df = pd.DataFrame(layout_rows)
    hough_lines_df = pd.DataFrame(hough_line_rows)
    hough_clusters_df = pd.DataFrame(hough_cluster_rows)
    timing_path = os.path.join(debug_dir, f"{video_name}_debug_timing.xlsx")
    result_path = os.path.join(debug_dir, f"{video_name}_debug_predictions.xlsx")
    layout_path = os.path.join(debug_dir, f"{video_name}_debug_channel_layout.xlsx")
    hough_lines_path = os.path.join(debug_dir, f"{video_name}_debug_hough_lines.xlsx")
    hough_clusters_path = os.path.join(debug_dir, f"{video_name}_debug_hough_clusters.xlsx")
    timing_df.to_excel(timing_path, index=False)
    result_df.to_excel(result_path, index=False)
    layout_df.to_excel(layout_path, index=False)
    hough_lines_df.to_excel(hough_lines_path, index=False)
    hough_clusters_df.to_excel(hough_clusters_path, index=False)

    if not result_df.empty:
        summary_df = build_channel_summary(result_df)
        summary_path = os.path.join(debug_dir, f"{video_name}_debug_channel_summary.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"Saved channel summary to: {summary_path}")

    if not timing_df.empty:
        print(f"Average sampled-frame time: {timing_df['elapsed_s'].mean():.3f}s")
        print(f"Min/Max sampled-frame time: {timing_df['elapsed_s'].min():.3f}s / {timing_df['elapsed_s'].max():.3f}s")

    print(f"Saved debug files to: {debug_dir}")
    return result_df


def main():
    start_time = time.time()
    rng = random.Random(DEBUG_RANDOM_SEED)

    print("=" * 50)
    print("Random-frame debug runner")
    print("=" * 50)
    configure_worker_runtime()
    print_runtime_device("Debug")
    check_model_files()

    video_paths = VIDEO
    if DEBUG_VIDEO_LIMIT is not None:
        video_paths = video_paths[:DEBUG_VIDEO_LIMIT]

    if not video_paths:
        print("VIDEO list is empty, nothing to debug.")
        return

    print("Loading models once...")
    detection_model = YOLO(DETECTION_MODEL_PATH)
    detection_model.to(RUNTIME_DEVICE)
    classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

    all_rows = []
    for video_path in video_paths:
        debug_df = process_random_frames(video_path, detection_model, classification_model, rng)
        if not debug_df.empty:
            all_rows.append(debug_df)

    if all_rows:
        merged_df = pd.concat(all_rows, ignore_index=True)
        summary_path = os.path.join(os.getcwd(), "debug_random_frame_predictions.xlsx")
        merged_df.to_excel(summary_path, index=False)
        print(f"\nSaved merged debug prediction sheet to: {summary_path}")

    print(f"Total debug execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
