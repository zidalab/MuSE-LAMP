import argparse
import copy
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_MODEL_ID = "convnextv2_base.fcmae_ft_in22k_in1k"
MODEL_ID_FALLBACKS = [
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_base.fcmae_ft_in1k",
    "convnextv2_base",
]


# -----------------------------------------------------------------------------
# VSCode / IDE click-run configuration
# -----------------------------------------------------------------------------
# 直接在 VSCode 里点击 Run Python File 时，会使用下面这些配置。
# 如果你在终端里手动传入命令行参数，则命令行参数优先。
USE_IDE_CONFIG_WHEN_NO_CLI_ARGS = True

# "train": 训练模型并在结束后导出验证集 Grad-CAM
# "gradcam_only": 只对 IDE_GRADCAM_IMAGE 做单张 Grad-CAM
IDE_RUN_MODE = "train"

IDE_CONFIG = {
    # Dataset and outputs.
    "data_dir": "datasets_cls_EB",
    "output_dir": "convnextv2b_cls_runs",
    "run_name": "ConvNeXtV2-B",

    # Training.
    "epochs": 30,
    "patience": 12,
    "batch_size": 16,
    "workers": 4,
    "img_size": 224,
    "lr": 5e-5,
    "weight_decay": 5e-2,
    "label_smoothing": 0.02,
    "dropout": 0.15,
    "model_id": DEFAULT_MODEL_ID,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "freeze_backbone": False,
    "use_class_weight": True,
    "no_pretrained": False,
    "amp": True,

    # Conservative augmentation.
    "rotation_deg": 5.0,
    "translate": 0.02,
    "scale_min": 0.97,
    "scale_max": 1.03,
    "hflip_prob": 0.20,
    "vflip_prob": 0.20,
    "jitter_prob": 0.25,
    "brightness": 0.05,
    "contrast": 0.05,
    "blur_prob": 0.08,
    "blur_sigma_min": 0.1,
    "blur_sigma_max": 0.4,
    "fill": 255,

    # Grad-CAM.
    "gradcam_samples": 12,
    "gradcam_layer": None,
    "gradcam_alpha": 0.45,
    "skip_gradcam": False,
    "checkpoint": "convnextv2b_cls_runs/ConvNeXtV2-B/best.pt",
    "image": "",
    "target_class": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a conservative ConvNeXt V2-B binary classifier with Grad-CAM."
    )
    parser.add_argument("--data-dir", type=str, default="datasets_cls_EB")
    parser.add_argument("--output-dir", type=str, default="convnextv2b_cls_runs")
    parser.add_argument("--run-name", type=str, default="ConvNeXtV2-B")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")

    # Conservative augmentation knobs.
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.02)
    parser.add_argument("--scale-min", type=float, default=0.97)
    parser.add_argument("--scale-max", type=float, default=1.03)
    parser.add_argument("--hflip-prob", type=float, default=0.20)
    parser.add_argument("--vflip-prob", type=float, default=0.20)
    parser.add_argument("--jitter-prob", type=float, default=0.25)
    parser.add_argument("--brightness", type=float, default=0.05)
    parser.add_argument("--contrast", type=float, default=0.05)
    parser.add_argument("--blur-prob", type=float, default=0.08)
    parser.add_argument("--blur-sigma-min", type=float, default=0.1)
    parser.add_argument("--blur-sigma-max", type=float, default=0.4)
    parser.add_argument("--fill", type=int, default=255, help="Augmentation fill value for crop background.")

    # Grad-CAM.
    parser.add_argument("--gradcam-samples", type=int, default=12)
    parser.add_argument("--gradcam-layer", type=str, default=None)
    parser.add_argument("--gradcam-alpha", type=float, default=0.45)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--gradcam-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--target-class", type=int, default=None)

    if USE_IDE_CONFIG_WHEN_NO_CLI_ARGS and len(sys.argv) == 1:
        args = parser.parse_args([])
        for key, value in IDE_CONFIG.items():
            setattr(args, key, value)
        args.gradcam_only = IDE_RUN_MODE.lower() == "gradcam_only"
        print("Running with IDE_CONFIG from ConvNeXtV2-B.py")
        print(f"IDE_RUN_MODE: {IDE_RUN_MODE}")
        return args

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "ConvNeXt V2-B is provided through timm in this script. "
            "Please install it first, for example: pip install timm"
        ) from exc
    return timm


def count_images(root_dir):
    counts = {}
    root = Path(root_dir)
    for split in ["train", "val", "test"]:
        split_dir = root / split
        class_counts = {}
        if split_dir.exists():
            for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
                if class_dir.is_dir():
                    class_counts[class_dir.name] = sum(
                        1 for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
                    )
        counts[split] = class_counts
    return counts


def build_transforms(args):
    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomAffine(
            degrees=args.rotation_deg,
            translate=(args.translate, args.translate),
            scale=(args.scale_min, args.scale_max),
            interpolation=InterpolationMode.BILINEAR,
            fill=args.fill,
        ),
        transforms.RandomHorizontalFlip(p=args.hflip_prob),
        transforms.RandomVerticalFlip(p=args.vflip_prob),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=0.0,
                hue=0.0,
            )
        ], p=args.jitter_prob),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(args.blur_sigma_min, args.blur_sigma_max),
            )
        ], p=args.blur_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])
    return train_tfms, eval_tfms


def build_datasets(data_dir, train_tfms, eval_tfms):
    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tfms)
    test_dir = os.path.join(data_dir, "test")
    test_set = datasets.ImageFolder(test_dir, transform=eval_tfms) if os.path.isdir(test_dir) else None

    if len(train_set.classes) != 2:
        raise ValueError(f"Expected binary classification with 2 classes, got {train_set.classes}")
    if train_set.classes != val_set.classes:
        raise ValueError(f"train classes {train_set.classes} do not match val classes {val_set.classes}")
    if test_set is not None and train_set.classes != test_set.classes:
        raise ValueError(f"train classes {train_set.classes} do not match test classes {test_set.classes}")

    return train_set, val_set, test_set


def build_loaders(train_set, val_set, test_set, batch_size, workers):
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs) if test_set is not None else None
    return train_loader, val_loader, test_loader


def build_model(num_classes, dropout, pretrained, model_id=DEFAULT_MODEL_ID):
    timm = import_timm()
    tried = []
    model_ids = [model_id] + [m for m in MODEL_ID_FALLBACKS if m != model_id]

    for candidate in model_ids:
        try:
            model = timm.create_model(
                candidate,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            return model, candidate, pretrained
        except Exception as exc:
            tried.append(f"{candidate}: {exc}")
            if pretrained:
                try:
                    print(
                        f"Warning: failed to load pretrained {candidate} ({exc}). "
                        "Retrying with random initialization."
                    )
                    model = timm.create_model(
                        candidate,
                        pretrained=False,
                        num_classes=num_classes,
                        drop_rate=dropout,
                    )
                    return model, candidate, False
                except Exception as retry_exc:
                    tried.append(f"{candidate} without pretrained: {retry_exc}")

    raise RuntimeError("Could not create ConvNeXt V2-B model. Tried:\n" + "\n".join(tried))


def maybe_freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith(("head.", "classifier.", "fc.")):
            param.requires_grad = False


def build_loss(train_set, use_class_weight, label_smoothing, device):
    if not use_class_weight:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    targets = [sample[1] for sample in train_set.samples]
    num_classes = len(train_set.classes)
    class_counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Class weights: {class_weights.tolist()}")
    return nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)


def summarize_metrics(total_loss, total_samples, tp, tn, fp, fn):
    accuracy = (tp + tn) / max(total_samples, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    loss = total_loss / max(total_samples, 1)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def update_confusion(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return tp, tn, fp, fn


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        batch_tp, batch_tn, batch_fp, batch_fn = update_confusion(preds, labels)
        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn

    return summarize_metrics(total_loss, total_samples, tp, tn, fp, fn)


def save_checkpoint(save_path, model, class_names, args, epoch, best_val_f1, best_val_acc, model_id, pretrained_loaded):
    checkpoint = {
        "model_name": "convnextv2_base",
        "model_family": "convnextv2",
        "model_id": model_id,
        "task": "binary_classification",
        "num_classes": len(class_names),
        "class_names": class_names,
        "img_size": args.img_size,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "dropout": args.dropout,
        "pretrained": pretrained_loaded,
    }
    torch.save(checkpoint, save_path)


def save_confusion_matrix(save_path, metrics, class_names):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", f"pred_{class_names[0]}", f"pred_{class_names[1]}"])
        writer.writerow([f"true_{class_names[0]}", metrics["tn"], metrics["fp"]])
        writer.writerow([f"true_{class_names[1]}", metrics["fn"], metrics["tp"]])


def is_better(val_metrics, best_val_f1, best_val_acc):
    if val_metrics["f1"] > best_val_f1 + 1e-6:
        return True
    if abs(val_metrics["f1"] - best_val_f1) <= 1e-6 and val_metrics["accuracy"] > best_val_acc:
        return True
    return False


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_id = checkpoint.get("model_id", DEFAULT_MODEL_ID)
    dropout = float(checkpoint.get("dropout", 0.0))
    num_classes = int(checkpoint.get("num_classes", len(checkpoint.get("class_names", ["0", "1"]))))
    model, model_id, _ = build_model(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False,
        model_id=model_id,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def pil_to_display_array(path, img_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(image)


def preprocess_image(path, img_size, device):
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])
    image = Image.open(path).convert("RGB")
    tensor = eval_tfms(image).unsqueeze(0).to(device)
    return tensor


def find_gradcam_layer(model, layer_name=None):
    module_dict = dict(model.named_modules())
    if layer_name:
        if layer_name not in module_dict:
            available = [name for name, module in module_dict.items() if isinstance(module, nn.Conv2d)]
            raise ValueError(
                f"Grad-CAM layer '{layer_name}' not found. "
                f"Some Conv2d candidates: {available[-12:]}"
            )
        return module_dict[layer_name], layer_name

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module, name

    raise ValueError("No Conv2d layer found for Grad-CAM.")


class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer, self.layer_name = find_gradcam_layer(model, layer_name)
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = int(probs.argmax(dim=1).item())
        if target_class is None:
            target_class = predicted_class

        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), predicted_class, float(probs[0, predicted_class].item())


def colorize_heatmap(heatmap):
    try:
        import cv2

        heatmap_uint8 = np.uint8(255 * heatmap)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[:, :, ::-1]
    except ImportError:
        import matplotlib.cm as cm

        return np.uint8(cm.jet(heatmap)[..., :3] * 255)


def save_gradcam_overlay(image_path, output_path, model, device, img_size, layer_name=None, target_class=None, alpha=0.45):
    input_tensor = preprocess_image(image_path, img_size, device)
    display = pil_to_display_array(image_path, img_size).astype(np.float32)

    gradcam = GradCAM(model, layer_name=layer_name)
    try:
        heatmap, pred_class, confidence = gradcam.generate(input_tensor, target_class=target_class)
    finally:
        gradcam.close()

    heatmap_rgb = colorize_heatmap(heatmap).astype(np.float32)
    overlay = display * (1.0 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)
    return pred_class, confidence, gradcam.layer_name


def export_gradcam_samples(model, dataset, save_dir, device, img_size, samples, layer_name=None, alpha=0.45):
    if samples <= 0:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    selected = dataset.samples[:samples]
    layer_used = None
    for index, (image_path, true_class) in enumerate(selected):
        stem = Path(image_path).stem
        output_path = save_dir / f"{index:03d}_true{true_class}_{stem}.jpg"
        pred_class, confidence, layer_used = save_gradcam_overlay(
            image_path=image_path,
            output_path=output_path,
            model=model,
            device=device,
            img_size=img_size,
            layer_name=layer_name,
            target_class=None,
            alpha=alpha,
        )
        final_path = save_dir / f"{index:03d}_true{true_class}_pred{pred_class}_conf{confidence:.3f}_{stem}.jpg"
        if final_path.exists():
            final_path.unlink()
        output_path.rename(final_path)

    print(f"Grad-CAM samples saved to: {save_dir}")
    print(f"Grad-CAM layer used: {layer_used}")


def run_gradcam_only(args):
    if not args.checkpoint:
        raise ValueError("--gradcam-only requires --checkpoint")
    if not args.image:
        raise ValueError("--gradcam-only requires --image")

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    img_size = int(checkpoint.get("img_size", args.img_size))
    save_dir = Path(args.output_dir) / args.run_name / "gradcam_only"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{Path(args.image).stem}_gradcam.jpg"
    pred_class, confidence, layer_used = save_gradcam_overlay(
        image_path=args.image,
        output_path=output_path,
        model=model,
        device=device,
        img_size=img_size,
        layer_name=args.gradcam_layer,
        target_class=args.target_class,
        alpha=args.gradcam_alpha,
    )
    class_names = checkpoint.get("class_names", ["0", "1"])
    print(f"Saved Grad-CAM to: {output_path}")
    print(f"Layer: {layer_used}")
    print(f"Prediction: {pred_class} ({class_names[pred_class]}) confidence={confidence:.4f}")


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.gradcam_only:
        run_gradcam_only(args)
        return

    train_tfms, eval_tfms = build_transforms(args)
    train_set, val_set, test_set = build_datasets(args.data_dir, train_tfms, eval_tfms)
    train_loader, val_loader, test_loader = build_loaders(
        train_set, val_set, test_set, args.batch_size, args.workers
    )

    save_dir = Path(args.output_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.csv"

    device = torch.device(args.device)
    model, model_id, pretrained_loaded = build_model(
        num_classes=len(train_set.classes),
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        model_id=args.model_id,
    )
    model = model.to(device)

    if args.freeze_backbone:
        maybe_freeze_backbone(model)

    criterion = build_loss(train_set, args.use_class_weight, args.label_smoothing, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset_counts = count_images(args.data_dir)
    print(f"Using device: {device}")
    print(f"Classes: {train_set.classes}")
    print(f"Dataset counts: {dataset_counts}")
    print(f"Model id: {model_id}")
    print(f"Pretrained weights loaded: {pretrained_loaded}")
    print(f"AMP enabled: {use_amp}")

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_acc = -1.0
    epochs_without_improve = 0

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_specificity",
            "val_loss",
            "val_acc",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_specificity",
            "lr",
        ])

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, use_amp=False)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                train_metrics["specificity"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["specificity"],
                current_lr,
            ])
            f.flush()

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['f1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"lr={current_lr:.6f}"
            )

            if is_better(val_metrics, best_val_f1, best_val_acc):
                best_val_f1 = val_metrics["f1"]
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    save_dir / "best.pt",
                    model,
                    train_set.classes,
                    args,
                    epoch,
                    best_val_f1,
                    best_val_acc,
                    model_id,
                    pretrained_loaded,
                )
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if args.patience > 0 and epochs_without_improve >= args.patience:
                    print(
                        "Early stopping triggered: "
                        f"validation F1 did not improve for {args.patience} epochs."
                    )
                    break

    save_checkpoint(
        save_dir / "last.pt",
        model,
        train_set.classes,
        args,
        epoch,
        best_val_f1,
        best_val_acc,
        model_id,
        pretrained_loaded,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, use_amp=False)
    save_confusion_matrix(save_dir / "val_confusion_matrix.csv", final_val_metrics, train_set.classes)
    print(f"Best epoch: {best_epoch}")
    print(f"Validation metrics: {final_val_metrics}")

    if test_loader is not None:
        test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None, use_amp=False)
        save_confusion_matrix(save_dir / "test_confusion_matrix.csv", test_metrics, train_set.classes)
        print(f"Test metrics: {test_metrics}")

    if not args.skip_gradcam:
        export_gradcam_samples(
            model=model,
            dataset=val_set,
            save_dir=save_dir / "gradcam_val",
            device=device,
            img_size=args.img_size,
            samples=args.gradcam_samples,
            layer_name=args.gradcam_layer,
            alpha=args.gradcam_alpha,
        )

    print(f"Best checkpoint saved to: {save_dir / 'best.pt'}")
    print(f"Last checkpoint saved to: {save_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
