from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from model.cm_gru import CM_GRU


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return -10.0 * math.log10(max(mse, 1e-10))


def compute_msssim(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    score = ms_ssim(a, b, data_range=1.0).item()
    score_db = -10.0 * math.log10(max(1.0 - score, 1e-10))
    return score, score_db


def pad(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = x.size()
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    left = (new_w - width) // 2
    right = new_w - width - left
    top = (new_h - height) // 2
    bottom = new_h - height - top
    padded = F.pad(x, (left, right, top, bottom), mode="constant", value=0)
    return padded, (left, right, top, bottom)


def crop(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    return F.pad(x, (-left, -right, -top, -bottom))


def infer_prompt_index(filename: str) -> int:
    match = re.match(r"(\d+)", Path(filename).stem)
    if match is None:
        raise ValueError(
            f"Could not infer the prompt index from '{filename}'. "
            "Expected the file name to start with a numeric id."
        )
    return int(match.group(1)) - 1


def load_prompts(text_path: Path) -> list[str]:
    with text_path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def select_prompt(image_path: Path, text_path: Path, prompt_override: str | None) -> str:
    if prompt_override is not None:
        return prompt_override

    prompts = load_prompts(text_path)
    prompt_index = infer_prompt_index(image_path.name)
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(
            f"Prompt index {prompt_index + 1} inferred from '{image_path.name}' is outside "
            f"the available prompt range [1, {len(prompts)}]."
        )
    return prompts[prompt_index]


def normalize_lambda(value: str) -> tuple[str, str]:
    decimal = value.replace("_", ".")
    float_value = float(decimal)
    canonical_decimal = f"{float_value:g}"
    slug = canonical_decimal.replace(".", "_")
    return canonical_decimal, slug


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    if args.lambda_value is None:
        raise ValueError("Provide either --checkpoint or --lambda-value.")

    decimal, slug = normalize_lambda(args.lambda_value)
    outer_dir = Path(args.checkpoint_root) / f"mj_only_{slug}"
    exact_candidates = [
        outer_dir / f"{args.hidden_channels}_{decimal}" / "checkpoint_latest.pth.tar",
        outer_dir / f"{args.hidden_channels}_{args.lambda_value}" / "checkpoint_latest.pth.tar",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    recursive_matches = sorted(outer_dir.glob("**/checkpoint_latest.pth.tar"))
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if recursive_matches:
        matches = "\n".join(str(match) for match in recursive_matches)
        raise FileNotFoundError(
            f"Multiple checkpoint candidates matched lambda={args.lambda_value}. "
            f"Pass --checkpoint explicitly.\n{matches}"
        )
    raise FileNotFoundError(
        f"Could not find a checkpoint for lambda={args.lambda_value} under {outer_dir}. "
        "Pass --checkpoint if the checkpoint is stored elsewhere."
    )


def resolve_image(args: argparse.Namespace) -> Path:
    test_dir = Path(args.test_dir)
    if args.image is not None:
        image = Path(args.image)
        if image.is_file():
            return image
        candidate = test_dir / args.image
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Image does not exist: {args.image}")

    image_paths = sorted(
        child for child in test_dir.iterdir() if child.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {test_dir}")
    if args.image_index < 0 or args.image_index >= len(image_paths):
        raise IndexError(
            f"--image-index {args.image_index} is outside the test image range "
            f"[0, {len(image_paths) - 1}]."
        )
    return image_paths[args.image_index]


def bitstream_num_bytes(strings: list[list[bytes]]) -> int:
    return sum(len(chunk) for stream in strings for chunk in stream)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    image_tensor = x.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(image_tensor)


def save_side_by_side(original: Image.Image, reconstruction: Image.Image, output_path: Path) -> None:
    width, height = original.size
    canvas = Image.new("RGB", (width * 2, height), color=(255, 255, 255))
    canvas.paste(original, (0, 0))
    canvas.paste(reconstruction, (width, 0))
    canvas.save(output_path)


def make_output_stem(lambda_value: str | None, checkpoint: Path, image_path: Path) -> str:
    if lambda_value is not None:
        _, slug = normalize_lambda(lambda_value)
        checkpoint_label = f"lambda_{slug}"
    else:
        checkpoint_label = checkpoint.parent.parent.name
    return f"mj_{checkpoint_label}_{image_path.stem}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one MJ test image with CATC, save the decoded reconstruction, "
            "save the entropy-coded payload, and write image-level metrics."
        )
    )
    parser.add_argument("--lambda-value", type=str, default=None, help="MJ lambda value, e.g. 0.05.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--checkpoint-root", default="checkpoints", type=str)
    parser.add_argument("--test-dir", default="data/MJ/test", type=str)
    parser.add_argument("--image", default=None, type=str, help="Image path or filename in --test-dir.")
    parser.add_argument("--image-index", default=0, type=int, help="Sorted test image index.")
    parser.add_argument("--text-path", default="data/MJ/MJ.txt", type=str)
    parser.add_argument("--prompt", default=None, type=str, help="Override prompt text.")
    parser.add_argument("--output-dir", default="results/single_image", type=str)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--clip-model", default="ViT-B/32", type=str)
    parser.add_argument("--hidden-channels", default=128, type=int)
    parser.add_argument("--latent-channels", default=320, type=int)
    parser.add_argument("--hyper-channels", default=192, type=int)
    parser.add_argument("--text-dim", default=512, type=int)
    parser.add_argument("--num-slices", default=5, type=int)
    parser.add_argument("--pad-multiple", default=64, type=int)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args)
    image_path = resolve_image(args)
    text_path = Path(args.text_path)
    if args.prompt is None and not text_path.is_file():
        raise FileNotFoundError(f"Prompt text file does not exist: {text_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = make_output_stem(args.lambda_value, checkpoint, image_path)
    reconstruction_path = output_dir / f"{output_stem}_reconstruction.png"
    comparison_path = output_dir / f"{output_stem}_comparison.png"
    payload_path = output_dir / f"{output_stem}_compressed_payload.pt"
    metrics_path = output_dir / f"{output_stem}_metrics.json"

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Image: {image_path}")

    prompt = select_prompt(image_path, text_path, args.prompt)
    text_model, _ = clip.load(args.clip_model, device=device, jit=False)
    text_model.eval()

    model = CM_GRU(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        text_dim=args.text_dim,
        num_slices=args.num_slices,
    ).to(device)
    model.eval()

    loaded_checkpoint = torch.load(checkpoint, map_location=device)
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in loaded_checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.update()

    original_image = Image.open(image_path).convert("RGB")
    image = transforms.ToTensor()(original_image).to(device)
    x = image.unsqueeze(0)
    x_padded, padding = pad(x, args.pad_multiple)

    text_tokens = clip.tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
        text_features = text_model.encode_text(text_tokens).float()

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        out_enc = model.compress(x_padded, text_features)
        out_dec = model.decompress(out_enc["strings"], text_features, out_enc["shape"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    x_hat = crop(out_dec["x_hat"], padding).clamp_(0, 1)
    reconstruction = tensor_to_pil(x_hat)
    reconstruction.save(reconstruction_path)
    save_side_by_side(original_image, reconstruction, comparison_path)

    bitstream_bytes = bitstream_num_bytes(out_enc["strings"])
    prompt_bytes = len(prompt.encode("utf-8"))
    compressed_payload_bytes = bitstream_bytes + prompt_bytes
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    psnr = compute_psnr(x, x_hat)
    ms_ssim_raw, ms_ssim_db = compute_msssim(x, x_hat)

    torch.save(
        {
            "strings": out_enc["strings"],
            "shape": out_enc["shape"],
            "padding": padding,
            "prompt": prompt,
            "image_path": str(image_path),
            "checkpoint": str(checkpoint),
            "lambda_value": args.lambda_value,
            "image_size": original_image.size,
            "pad_multiple": args.pad_multiple,
        },
        payload_path,
    )

    metrics: dict[str, Any] = {
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "lambda_value": args.lambda_value,
        "prompt": prompt,
        "width": original_image.size[0],
        "height": original_image.size[1],
        "num_pixels": num_pixels,
        "original_file_size_bytes": image_path.stat().st_size,
        "bitstream_bytes": bitstream_bytes,
        "prompt_bytes": prompt_bytes,
        "compressed_payload_bytes": compressed_payload_bytes,
        "bitrate_bpp_including_prompt": compressed_payload_bytes * 8.0 / num_pixels,
        "bitrate_bpp_image_bitstream_only": bitstream_bytes * 8.0 / num_pixels,
        "psnr_db": psnr,
        "ms_ssim": ms_ssim_raw,
        "ms_ssim_db": ms_ssim_db,
        "elapsed_ms": elapsed * 1000.0,
        "reconstruction_path": str(reconstruction_path),
        "comparison_path": str(comparison_path),
        "compressed_payload_path": str(payload_path),
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
