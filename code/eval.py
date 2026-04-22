from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from model.cm_gru import CM_GRU


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return -10.0 * math.log10(max(mse, 1e-10))


def compute_msssim(a: torch.Tensor, b: torch.Tensor) -> float:
    score = ms_ssim(a, b, data_range=1.0).item()
    return -10.0 * math.log10(max(1.0 - score, 1e-10))


def pad(x: torch.Tensor, multiple: int):
    _, _, height, width = x.size()
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    left = (new_w - width) // 2
    right = new_w - width - left
    top = (new_h - height) // 2
    bottom = new_h - height - top
    padded = F.pad(x, (left, right, top, bottom), mode="constant", value=0)
    return padded, (left, right, top, bottom)


def crop(x: torch.Tensor, padding):
    left, right, top, bottom = padding
    return F.pad(x, (-left, -right, -top, -bottom))


def infer_prompt_index(filename: str) -> int:
    match = re.match(r"(\d+)", Path(filename).stem)
    if match is None:
        raise ValueError(f"Could not infer prompt index from '{filename}'.")
    return int(match.group(1)) - 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluate CATC on image / text pairs.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-i", required=True, type=str, help="Directory with images")
    parser.add_argument("--data-t", required=True, type=str, help="Text file with prompts")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--clip-model", default="ViT-B/32", type=str)
    parser.add_argument("--hidden-channels", default=128, type=int)
    parser.add_argument("--latent-channels", default=320, type=int)
    parser.add_argument("--hyper-channels", default=192, type=int)
    parser.add_argument("--text-dim", default=512, type=int)
    parser.add_argument("--num-slices", default=5, type=int)
    parser.add_argument("--pad-multiple", default=64, type=int)
    return parser.parse_args(argv)


def main(argv) -> None:
    args = parse_args(argv)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = {
        key.replace("module.", "", 1): value for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.update()

    image_dir = Path(args.data_i)
    with Path(args.data_t).open("r", encoding="utf-8") as handle:
        prompts = [line.rstrip("\n") for line in handle]

    image_names = sorted(
        name
        for name in os.listdir(image_dir)
        if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    total_psnr = 0.0
    total_ms_ssim = 0.0
    total_bpp = 0.0
    total_time = 0.0

    for image_name in image_names:
        prompt_index = infer_prompt_index(image_name)
        prompt = prompts[prompt_index]
        text_tokens = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            text_features = text_model.encode_text(text_tokens).float()

        image = transforms.ToTensor()(Image.open(image_dir / image_name).convert("RGB")).to(device)
        x = image.unsqueeze(0)
        x_padded, padding = pad(x, args.pad_multiple)

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
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        text_bytes = len(prompt.encode("utf-8"))
        bitstream_bytes = len(out_enc["strings"][0][0]) + sum(len(chunk) for chunk in out_enc["strings"][1])
        bitrate = (bitstream_bytes + text_bytes) * 8.0 / num_pixels

        psnr = compute_psnr(x, x_hat)
        ms_ssim_db = compute_msssim(x, x_hat)

        print(f"{image_name}: bpp={bitrate:.4f} psnr={psnr:.3f}dB ms-ssim={ms_ssim_db:.3f}dB")

        total_bpp += bitrate
        total_psnr += psnr
        total_ms_ssim += ms_ssim_db
        total_time += elapsed

    count = max(len(image_names), 1)
    print(f"average_PSNR: {total_psnr / count:.2f}dB")
    print(f"average_MS-SSIM: {total_ms_ssim / count:.4f}dB")
    print(f"average_Bit-rate: {total_bpp / count:.4f} bpp")
    print(f"average_time: {1000.0 * total_time / count:.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1:])
