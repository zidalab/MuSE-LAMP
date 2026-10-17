"""Classify images and visualize the classifier's attention with Grad-CAM.

Examples
--------
python gradcam_cls.py --image data/example.png
python gradcam_cls.py --image data/image1.png data/image2.png
python gradcam_cls.py --image data/images
python gradcam_cls.py --image data/example.png --target-class 1
python gradcam_cls.py --image data/example.png --layer stages.3.blocks.2.conv_dw

The script is written for the ConvNeXt V2 checkpoint produced by
``train_ConvNeXtV2-B_cls.py``.  Model metadata (class names, image size and
normalization statistics) is read directly from the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "convnextv2b_cls_runs" / "ConvNeXtV2-B" / "best.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures" / "gradcam_cls"

# If you normally click "Run Python File" in VS Code, put one of these here:
#   - one image:  "E:/data/image.png"
#   - a folder:   "E:/data/images" (all images below it are processed recursively)
#   - many paths: ["E:/data/image1.png", "E:/data/image2.png", "E:/data/images"]
# Command-line --image takes precedence over this value.
IDE_IMAGE_PATH: str | Path | Sequence[str | Path] = (
    "E:/VsCode Project/DeepLamp_4channel/display droplet/positive"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_MODEL_ID = "convnextv2_base.fcmae_ft_in22k_in1k"
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def get_ide_image_paths() -> list[Path] | None:
    if not IDE_IMAGE_PATH:
        return None
    if isinstance(IDE_IMAGE_PATH, (str, Path)):
        return [Path(IDE_IMAGE_PATH)]
    return [Path(path) for path in IDE_IMAGE_PATH]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one or more images and save their Grad-CAM heatmaps."
    )
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=get_ide_image_paths(),
        help=(
            "One or more input images or folders. Folders are searched recursively. "
            "Command-line values override IDE_IMAGE_PATH."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Classification checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file for one image, or output directory for multiple images. "
            "By default overlays are saved under figures/gradcam_cls/."
        ),
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="1",
        help=(
            "Class index or class name to explain (default: 1, the positive/precipitate class). "
            "Use 'predicted' to explain whichever class the model predicts."
        ),
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Target layer name. By default, use the last Conv2d layer.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity in the overlay, between 0 and 1 (default: 0.45).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap used for the heatmap (default: jet).",
    )
    args = parser.parse_args()

    if args.image is None:
        parser.error(
            "Please pass --image IMAGE_OR_FOLDER, or set IDE_IMAGE_PATH at the top of the script."
        )
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")
    return args


def collect_image_paths(inputs: Sequence[Path]) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[str] = set()

    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            if not candidates:
                raise ValueError(f"No supported images found in folder: {path}")
        else:
            raise FileNotFoundError(f"Image or folder not found: {path}")

        for candidate in candidates:
            key = str(candidate).casefold()
            if key not in seen:
                seen.add(key)
                image_paths.append(candidate)

    if not image_paths:
        raise ValueError("No input images were found.")
    return image_paths


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False is required for full training checkpoints on PyTorch 2.6+.
    # Fall back for older PyTorch versions which do not expose this argument.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format: expected a dictionary containing 'state_dict'."
        )
    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("This script requires timm. Install it with: pip install timm") from exc

    class_names = checkpoint.get("class_names", ["0", "1"])
    num_classes = int(checkpoint.get("num_classes", len(class_names)))
    model_id = str(checkpoint.get("model_id", DEFAULT_MODEL_ID))
    dropout = float(checkpoint.get("dropout", 0.0))

    model = timm.create_model(
        model_id,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(
    image: Image.Image,
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def find_target_layer(model: nn.Module, requested_name: str | None) -> tuple[str, nn.Module]:
    named_modules = dict(model.named_modules())

    if requested_name is not None:
        if requested_name not in named_modules:
            candidates = [
                name for name, module in named_modules.items() if isinstance(module, nn.Conv2d)
            ]
            candidate_text = ", ".join(candidates[-12:])
            raise ValueError(
                f"Layer '{requested_name}' was not found. Last Conv2d candidates: {candidate_text}"
            )
        return requested_name, named_modules[requested_name]

    # ConvNeXt's depthwise convolution is only an intermediate operation inside
    # a block. Hooking it can emphasize padding/background before the pointwise
    # MLP and residual fusion. The complete final ConvNeXt block gives a much
    # more faithful class activation map.
    for name, module in reversed(list(model.named_modules())):
        if module.__class__.__name__ == "ConvNeXtBlock":
            return name, module

    # Generic fallback for conventional CNN classifiers.
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return name, module
    raise ValueError("No Conv2d layer was found in this model.")


class GradCAM:
    """Minimal Grad-CAM implementation for a CNN classifier."""

    def __init__(self, model: nn.Module, layer_name: str | None = None) -> None:
        self.model = model
        self.layer_name, self.target_layer = find_target_layer(model, layer_name)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("The selected Grad-CAM layer must output a tensor.")
        self.activations = output
        # The prediction-only pass runs under inference_mode(), where tensors do
        # not require gradients. The later Grad-CAM pass does and gets the hook.
        if output.requires_grad:
            output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def close(self) -> None:
        self._handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shaped [batch, classes], got {tuple(logits.shape)}")
        if not 0 <= target_class < logits.shape[1]:
            raise ValueError(
                f"Target class {target_class} is outside the valid range 0..{logits.shape[1] - 1}."
            )

        logits[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations or gradients.")
        if self.activations.ndim != 4 or self.gradients.ndim != 4:
            raise ValueError(
                "The selected layer must have BCHW-shaped activations. "
                f"Got {tuple(self.activations.shape)}."
            )

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((channel_weights * activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap -= heatmap.min()
        peak = heatmap.max()
        if peak > 0:
            heatmap /= peak

        probabilities = torch.softmax(logits.detach(), dim=1)[0]
        return heatmap.cpu().numpy(), probabilities.cpu()


def resolve_target_class(
    value: str | None,
    predicted_class: int,
    class_names: Sequence[str],
) -> int:
    if value is None or value.strip().lower() in {"predicted", "prediction", "pred", "auto"}:
        return predicted_class

    try:
        target_class = int(value)
    except ValueError:
        exact_matches = [index for index, name in enumerate(class_names) if name == value]
        if not exact_matches:
            raise ValueError(
                f"Unknown target class '{value}'. Available classes: {list(class_names)}"
            ) from None
        target_class = exact_matches[0]

    if not 0 <= target_class < len(class_names):
        raise ValueError(f"Target class must be in 0..{len(class_names) - 1}.")
    return target_class


def resize_heatmap_to_image(heatmap: np.ndarray, image: Image.Image) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L")
    heatmap_image = heatmap_image.resize(image.size, resample=Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image, dtype=np.float32) / 255.0


def colorize_heatmap(heatmap: np.ndarray, colormap_name: str) -> np.ndarray:
    import matplotlib

    try:
        colormap = matplotlib.colormaps[colormap_name]
    except KeyError:
        available = ", ".join(list(matplotlib.colormaps)[:20])
        raise ValueError(
            f"Unknown colormap '{colormap_name}'. Examples: {available}"
        ) from None
    return np.uint8(colormap(np.clip(heatmap, 0.0, 1.0))[..., :3] * 255)


def save_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float,
    colormap_name: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        output_path = output_path.with_suffix(".png")

    full_heatmap = resize_heatmap_to_image(heatmap, image)
    heatmap_rgb = colorize_heatmap(full_heatmap, colormap_name)
    image_array = np.asarray(image, dtype=np.float32)
    overlay = np.clip(
        image_array * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)
    return output_path


def choose_output_path(
    image_path: Path,
    requested_output: Path | None,
    is_batch: bool,
    used_paths: set[str],
) -> Path:
    filename = f"{image_path.stem}_gradcam_overlay.png"
    if requested_output is None:
        candidate = DEFAULT_OUTPUT_DIR / filename
    else:
        requested_output = requested_output.expanduser().resolve()
        if is_batch:
            if requested_output.exists() and requested_output.is_file():
                raise ValueError(
                    f"--output must be a directory when processing multiple images: {requested_output}"
                )
            if requested_output.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                raise ValueError(
                    "--output must be a directory, not an image filename, when processing multiple images."
                )
            candidate = requested_output / filename
        elif requested_output.is_dir():
            candidate = requested_output / filename
        else:
            candidate = requested_output

    # Images from different input folders can share a stem. Avoid overwriting
    # an earlier result from the same batch in that case.
    base_candidate = candidate
    duplicate_index = 2
    while str(candidate).casefold() in used_paths:
        candidate = base_candidate.with_name(
            f"{base_candidate.stem}_{duplicate_index}{base_candidate.suffix}"
        )
        duplicate_index += 1
    used_paths.add(str(candidate).casefold())
    return candidate


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.image)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = choose_device(args.device)

    checkpoint = load_checkpoint(checkpoint_path)
    class_names = [str(name) for name in checkpoint.get("class_names", ["0", "1"])]
    img_size = int(checkpoint.get("img_size", 224))
    mean = tuple(float(value) for value in checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(float(value) for value in checkpoint.get("std", DEFAULT_STD))

    model = build_model(checkpoint, device)
    gradcam = GradCAM(model, args.layer)
    is_batch = len(image_paths) > 1
    used_output_paths: set[str] = set()
    succeeded = 0
    failures: list[tuple[Path, str]] = []

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Grad-CAM layer: {gradcam.layer_name}")
    print(f"Images: {len(image_paths)}")

    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            try:
                image = load_rgb_image(image_path)
                input_tensor = preprocess_image(image, img_size, mean, std, device)

                # A first gradient-free pass determines which class should be
                # explained when --target-class is set to "predicted".
                with torch.inference_mode():
                    initial_logits = model(input_tensor)
                    initial_probabilities = torch.softmax(initial_logits, dim=1)[0]
                predicted_class = int(initial_probabilities.argmax().item())
                target_class = resolve_target_class(
                    args.target_class, predicted_class, class_names
                )
                heatmap, probabilities = gradcam.generate(input_tensor, target_class)

                predicted_class = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted_class].item())
                output_path = choose_output_path(
                    image_path=image_path,
                    requested_output=args.output,
                    is_batch=is_batch,
                    used_paths=used_output_paths,
                )
                overlay_path = save_overlay(
                    image=image,
                    heatmap=heatmap,
                    output_path=output_path,
                    alpha=args.alpha,
                    colormap_name=args.colormap,
                )

                print(f"\n[{image_index}/{len(image_paths)}] Image: {image_path}")
                print(f"Prediction: {predicted_class} ({class_names[predicted_class]})")
                print(f"Confidence: {confidence:.4f}")
                print(f"Explained class: {target_class} ({class_names[target_class]})")
                print("Class probabilities:")
                for index, (class_name, probability) in enumerate(
                    zip(class_names, probabilities.tolist())
                ):
                    print(f"  {index}: {class_name} = {probability:.4f}")
                print(f"Overlay: {overlay_path}")
                succeeded += 1
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append((image_path, str(exc)))
                print(
                    f"\n[{image_index}/{len(image_paths)}] Failed: {image_path}\n"
                    f"  {exc}",
                    file=sys.stderr,
                )
    finally:
        gradcam.close()

    print(f"\nDone: {succeeded} succeeded, {len(failures)} failed.")
    if failures:
        failed_names = "\n".join(f"  - {path}: {message}" for path, message in failures)
        raise RuntimeError(f"Some images could not be processed:\n{failed_names}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
