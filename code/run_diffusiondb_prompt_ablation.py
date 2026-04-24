from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import clip
import torch
import torch.optim as optim
from PIL import Image
from remotezip import RemoteZip
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval import compute_msssim, compute_psnr, crop, pad
from model.cm_gru import CM_GRU
from train import (
    RateDistortionLoss,
    build_checkpoint_payload,
    configure_optimizers,
    encode_text_batch,
    save_checkpoint,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROMPT_CASES = ("correct", "empty", "shuffled_words", "swapped_prompts")
REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSIONDB_PART_COUNTS = {"2m": 2000, "large": 14000}


class LocalPromptDataset(Dataset):
    def __init__(self, data_root: Path, records: list[dict[str, Any]], image_transform) -> None:
        self.data_root = data_root
        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = self.data_root / record["image_path"]
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image), record["prompt"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Download a small DiffusionDB subset directly from the official remote shard zips, "
            "train CATC on correct prompts, and evaluate prompt-corruption cases."
        )
    )
    parser.add_argument("--dataset-repo", default="poloclub/diffusiondb", type=str)
    parser.add_argument(
        "--diffusiondb-subset",
        default="2m",
        choices=["2m", "large"],
        help="Which official DiffusionDB image collection to sample from.",
    )
    parser.add_argument(
        "--part-ids",
        nargs="*",
        default=None,
        type=int,
        help=(
            "Optional shard ids to sample from. If omitted, enough shards are selected "
            "deterministically from the seed to cover --sample-count."
        ),
    )
    parser.add_argument("--sample-count", default=50, type=int)
    parser.add_argument("--train-count", default=40, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--aux-learning-rate", default=5e-4, type=float)
    parser.add_argument("--lambda", dest="lmbda", default=0.05, type=float)
    parser.add_argument("--metric", default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--patch-size", nargs=2, default=(256, 256), type=int)
    parser.add_argument("--clip-max-norm", default=1.0, type=float)
    parser.add_argument("--lr-epoch", nargs="*", default=[45, 48], type=int)
    parser.add_argument("--clip-model", default="ViT-B/32", type=str)
    parser.add_argument("--hidden-channels", default=16, type=int)
    parser.add_argument("--latent-channels", default=40, type=int)
    parser.add_argument("--hyper-channels", default=24, type=int)
    parser.add_argument("--text-dim", default=512, type=int)
    parser.add_argument("--num-slices", default=5, type=int)
    parser.add_argument("--pad-multiple", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--workspace-dir",
        default=str(REPO_ROOT / "diffusiondb_prompt_ablation"),
        type=str,
        help="Directory used for downloaded data, metadata, and checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results"),
        type=str,
        help="Directory where per-case JSON result arrays will be written.",
    )
    parser.add_argument(
        "--experiment-name",
        default="diffusiondb_prompt_ablation",
        type=str,
        help="Prefix used for result files and saved manifests.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Optional checkpoint path. If provided, training is skipped and this checkpoint is used.",
    )
    parser.add_argument(
        "--save-training-state",
        action="store_true",
        help="Include optimizer and scheduler state in checkpoints.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download the DiffusionDB subset even if a matching local manifest exists.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if a local checkpoint already exists.",
    )
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.sample_count < 2:
        raise ValueError("--sample-count must be at least 2.")
    if args.train_count < 1:
        raise ValueError("--train-count must be at least 1.")
    if args.train_count >= args.sample_count:
        raise ValueError("--train-count must be smaller than --sample-count.")
    max_parts = DIFFUSIONDB_PART_COUNTS[args.diffusiondb_subset]
    if args.sample_count > max_parts * 1000:
        raise ValueError(
            f"--sample-count {args.sample_count} exceeds the available capacity of the "
            f"{args.diffusiondb_subset} subset."
        )
    if args.part_ids is not None:
        for part_id in args.part_ids:
            if part_id < 1 or part_id > max_parts:
                raise ValueError(
                    f"Part id {part_id} is outside the valid range [1, {max_parts}] "
                    f"for the {args.diffusiondb_subset} subset."
                )


def resolve_part_ids(subset: str, requested_part_ids: Optional[list[int]], sample_count: int, seed: int) -> list[int]:
    if requested_part_ids:
        return requested_part_ids

    num_parts = math.ceil(sample_count / 1000)
    max_parts = DIFFUSIONDB_PART_COUNTS[subset]
    rng = random.Random(seed)
    return sorted(rng.sample(range(1, max_parts + 1), num_parts))


def build_zip_relative_path(subset: str, part_id: int) -> str:
    shard_name = f"part-{part_id:06d}.zip"
    if subset == "2m":
        return f"images/{shard_name}"
    if part_id <= 10000:
        return f"diffusiondb-large-part-1/{shard_name}"
    return f"diffusiondb-large-part-2/{shard_name}"


def build_zip_url(dataset_repo: str, subset: str, part_id: int) -> str:
    relative_path = build_zip_relative_path(subset, part_id)
    return f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{relative_path}"


def build_split_records(
    records: list[dict[str, Any]],
    train_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_indices = set(indices[:train_count])
    train_records = [record for idx, record in enumerate(records) if idx in train_indices]
    eval_records = [record for idx, record in enumerate(records) if idx not in train_indices]
    return train_records, eval_records


def load_existing_manifest(
    manifest_path: Path,
    args,
    part_ids: list[int],
) -> Optional[dict[str, Any]]:
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected = {
        "dataset_repo": args.dataset_repo,
        "diffusiondb_subset": args.diffusiondb_subset,
        "part_ids": part_ids,
        "sample_count": args.sample_count,
        "train_count": args.train_count,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None

    data_root = manifest_path.parent
    for record in manifest.get("records", []):
        if not (data_root / record["image_path"]).is_file():
            return None
    return manifest


def allocate_samples(sample_count: int, num_parts: int) -> list[int]:
    base = sample_count // num_parts
    remainder = sample_count % num_parts
    return [base + (1 if idx < remainder else 0) for idx in range(num_parts)]


def load_shard_records(zip_url: str, part_id: int) -> dict[str, Any]:
    json_name = f"part-{part_id:06d}.json"
    with RemoteZip(zip_url) as archive:
        names = set(archive.namelist())
        if json_name not in names:
            raise FileNotFoundError(f"Could not locate {json_name} inside {zip_url}.")
        with archive.open(json_name) as handle:
            shard_metadata = json.load(handle)
    return shard_metadata


def sample_shard_records(
    shard_metadata: dict[str, Any],
    sample_count: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    items = list(shard_metadata.items())
    if sample_count > len(items):
        raise ValueError(
            f"Requested {sample_count} samples from a shard that only exposes {len(items)} records."
        )
    rng = random.Random(seed)
    return rng.sample(items, sample_count)


def save_selected_images(
    zip_url: str,
    selected_items: list[tuple[str, dict[str, Any]]],
    images_dir: Path,
    local_id_offset: int,
    part_id: int,
) -> list[dict[str, Any]]:
    records = []
    with RemoteZip(zip_url) as archive:
        for offset, (image_name, metadata) in enumerate(selected_items, start=1):
            local_id = local_id_offset + offset
            local_name = f"{local_id:04d}.png"
            relative_path = Path("images") / local_name
            image_path = images_dir.parent / relative_path

            with archive.open(image_name) as handle:
                image = Image.open(handle).convert("RGB")
                image.save(image_path, format="PNG")

            records.append(
                {
                    "local_id": local_id,
                    "image_path": relative_path.as_posix(),
                    "image_name": local_name,
                    "source_image_name": image_name,
                    "prompt": str(metadata.get("p", "")),
                    "part_id": part_id,
                    "source_seed": metadata.get("seed"),
                    "width": int(image.width),
                    "height": int(image.height),
                }
            )
    return records


def download_subset(args, data_root: Path, part_ids: list[int]) -> dict[str, Any]:
    data_root.mkdir(parents=True, exist_ok=True)
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.json"

    if not args.force_redownload:
        cached = load_existing_manifest(manifest_path, args, part_ids)
        if cached is not None:
            print(f"Reusing cached DiffusionDB subset from {manifest_path}")
            return cached

    sample_plan = allocate_samples(args.sample_count, len(part_ids))
    records = []
    local_id_offset = 0
    shard_summaries = []

    for shard_seed, (part_id, shard_sample_count) in enumerate(zip(part_ids, sample_plan), start=1):
        zip_url = build_zip_url(args.dataset_repo, args.diffusiondb_subset, part_id)
        print(
            f"Sampling {shard_sample_count} pairs from "
            f"{args.diffusiondb_subset} shard part-{part_id:06d}"
        )
        shard_metadata = load_shard_records(zip_url, part_id)
        selected_items = sample_shard_records(
            shard_metadata,
            sample_count=shard_sample_count,
            seed=args.seed + shard_seed,
        )
        shard_records = save_selected_images(
            zip_url=zip_url,
            selected_items=selected_items,
            images_dir=images_dir,
            local_id_offset=local_id_offset,
            part_id=part_id,
        )
        local_id_offset += len(shard_records)
        records.extend(shard_records)
        shard_summaries.append(
            {
                "part_id": part_id,
                "zip_url": zip_url,
                "sample_count": shard_sample_count,
            }
        )

    train_records, eval_records = build_split_records(records, args.train_count, args.seed)
    manifest = {
        "dataset_repo": args.dataset_repo,
        "diffusiondb_subset": args.diffusiondb_subset,
        "part_ids": part_ids,
        "sample_count": args.sample_count,
        "train_count": args.train_count,
        "eval_count": len(eval_records),
        "seed": args.seed,
        "records": records,
        "train_records": train_records,
        "eval_records": eval_records,
        "shards": shard_summaries,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_train_transform(patch_size: tuple[int, int]):
    resize_size = max(patch_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.RandomCrop(tuple(patch_size)),
            transforms.ToTensor(),
        ]
    )


def build_model(args, device: torch.device) -> CM_GRU:
    return CM_GRU(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        text_dim=args.text_dim,
        num_slices=args.num_slices,
    ).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        key.replace("module.", "", 1): value for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)


def train_model(
    args,
    train_records: list[dict[str, Any]],
    data_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
    text_model,
) -> tuple[CM_GRU, Path]:
    checkpoint_path = checkpoint_dir / f"{args.hidden_channels}_{args.lmbda}" / "checkpoint_latest.pth.tar"
    model = build_model(args, device)

    if args.checkpoint:
        explicit_checkpoint = Path(args.checkpoint)
        print(f"Using provided checkpoint: {explicit_checkpoint}")
        load_checkpoint(model, explicit_checkpoint, device)
        model.update()
        model.eval()
        return model, explicit_checkpoint

    if checkpoint_path.is_file() and not args.force_retrain:
        print(f"Reusing local checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, device)
        model.update()
        model.eval()
        return model, checkpoint_path

    train_dataset = LocalPromptDataset(
        data_root=data_root,
        records=train_records,
        image_transform=build_train_transform(tuple(args.patch_size)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer, aux_optimizer = configure_optimizers(model, args)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_epoch, gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)
    save_dir = checkpoint_path.parent
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, (images, prompts) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            text_features = encode_text_batch(text_model, prompts, device)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            out_net = model(images, text_features)
            out_loss = criterion(out_net, images)
            out_loss["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            running_loss += float(out_loss["loss"].item())
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                distortion_key = "mse_loss" if args.metric == "mse" else "ms_ssim_loss"
                print(
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={out_loss['loss'].item():.4f} "
                    f"{distortion_key}={out_loss[distortion_key].item():.6f} "
                    f"bpp={out_loss['bpp_loss'].item():.4f} "
                    f"aux={aux_loss.item():.4f}"
                )

        scheduler.step()
        epoch_loss = running_loss / max(len(train_loader), 1)
        best_loss = min(best_loss, epoch_loss)
        checkpoint = build_checkpoint_payload(
            model=model,
            epoch=epoch,
            loss=epoch_loss,
            best_loss=best_loss,
            args=args,
            include_training_state=args.save_training_state,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            scheduler=scheduler,
        )
        save_checkpoint(checkpoint, epoch, save_dir, save_epoch_checkpoints=False)
        print(f"Finished epoch {epoch} with average training loss {epoch_loss:.4f}")

    model.update()
    model.eval()
    return model, checkpoint_path


def shuffle_prompt_words(prompt: str, seed: int) -> str:
    words = prompt.split()
    if len(words) <= 1:
        return prompt
    rng = random.Random(seed)
    shuffled = words[:]
    for _ in range(8):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)


def build_deranged_prompts(prompts: list[str], seed: int) -> list[str]:
    if len(prompts) <= 1:
        return prompts[:]

    indices = list(range(len(prompts)))
    rng = random.Random(seed)
    for _ in range(128):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(old != new for old, new in zip(indices, shuffled)):
            return [prompts[idx] for idx in shuffled]

    shift = rng.randrange(1, len(prompts))
    return [prompts[(idx + shift) % len(prompts)] for idx in indices]


def build_prompt_cases(eval_records: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    prompts = [record["prompt"] for record in eval_records]
    return {
        "correct": prompts,
        "empty": ["" for _ in prompts],
        "shuffled_words": [
            shuffle_prompt_words(prompt, seed + idx + 1) for idx, prompt in enumerate(prompts)
        ],
        "swapped_prompts": build_deranged_prompts(prompts, seed + 10_000),
    }


def evaluate_case(
    args,
    case_name: str,
    prompts: list[str],
    eval_records: list[dict[str, Any]],
    data_root: Path,
    model: CM_GRU,
    text_model,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    results = []

    with torch.no_grad():
        for record, prompt in zip(eval_records, prompts):
            image_path = data_root / record["image_path"]
            original_prompt = record["prompt"]

            text_tokens = clip.tokenize([prompt], truncate=True).to(device)
            text_features = text_model.encode_text(text_tokens).float()

            image = transforms.ToTensor()(Image.open(image_path).convert("RGB")).to(device)
            x = image.unsqueeze(0)
            x_padded, padding = pad(x, args.pad_multiple)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            out_enc = model.compress(x_padded, text_features)
            out_dec = model.decompress(out_enc["strings"], text_features, out_enc["shape"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = 1000.0 * (time.time() - start)

            x_hat = crop(out_dec["x_hat"], padding).clamp_(0, 1)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            image_bitstream = len(out_enc["strings"][0][0]) + sum(
                len(chunk) for chunk in out_enc["strings"][1]
            )
            prompt_text_size = len(prompt.encode("utf-8"))
            compressed_size = image_bitstream + prompt_text_size
            original_text_size = len(original_prompt.encode("utf-8"))

            results.append(
                {
                    "prompt_case": case_name,
                    "local_id": record["local_id"],
                    "image_name": record["image_name"],
                    "image_path": str((data_root / record["image_path"]).resolve()),
                    "source_image_name": record["source_image_name"],
                    "part_id": record["part_id"],
                    "prompt": prompt,
                    "original_prompt": original_prompt,
                    "original_image_size": image_path.stat().st_size,
                    "original_text_size": original_text_size,
                    "prompt_text_size": prompt_text_size,
                    "image_bitstream": image_bitstream,
                    "compressed_size": compressed_size,
                    "bpp": compressed_size * 8.0 / num_pixels,
                    "psnr_db": compute_psnr(x, x_hat),
                    "ms_ssim_db": compute_msssim(x, x_hat),
                    "elapsed_ms": elapsed_ms,
                }
            )

    return results


def summarize_case(results: list[dict[str, Any]]) -> dict[str, float]:
    summary = {
        "count": float(len(results)),
        "average_image_bitstream": 0.0,
        "average_compressed_size": 0.0,
        "average_original_image_size": 0.0,
        "average_original_text_size": 0.0,
        "average_prompt_text_size": 0.0,
        "average_bpp": 0.0,
        "average_psnr_db": 0.0,
        "average_ms_ssim_db": 0.0,
        "average_elapsed_ms": 0.0,
    }
    if not results:
        return summary

    count = float(len(results))
    for key, result_key in (
        ("average_image_bitstream", "image_bitstream"),
        ("average_compressed_size", "compressed_size"),
        ("average_original_image_size", "original_image_size"),
        ("average_original_text_size", "original_text_size"),
        ("average_prompt_text_size", "prompt_text_size"),
        ("average_bpp", "bpp"),
        ("average_psnr_db", "psnr_db"),
        ("average_ms_ssim_db", "ms_ssim_db"),
        ("average_elapsed_ms", "elapsed_ms"),
    ):
        summary[key] = sum(float(result[result_key]) for result in results) / count
    return summary


def save_results(
    args,
    results_dir: Path,
    case_results: dict[str, list[dict[str, Any]]],
    summary: dict[str, dict[str, float]],
    manifest: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for case_name, results in case_results.items():
        write_json(results_dir / f"{args.experiment_name}_{case_name}.json", results)

    write_json(results_dir / f"{args.experiment_name}_summary.json", summary)
    write_json(
        results_dir / f"{args.experiment_name}_manifest.json",
        {
            "experiment_name": args.experiment_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "dataset_repo": manifest["dataset_repo"],
            "diffusiondb_subset": manifest["diffusiondb_subset"],
            "part_ids": manifest["part_ids"],
            "sample_count": manifest["sample_count"],
            "train_count": manifest["train_count"],
            "eval_count": manifest["eval_count"],
            "cases": list(case_results.keys()),
        },
    )
    write_json(results_dir / f"{args.experiment_name}_all_cases.json", case_results)


def main(argv) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed)

    part_ids = resolve_part_ids(
        subset=args.diffusiondb_subset,
        requested_part_ids=args.part_ids,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    workspace_dir = Path(args.workspace_dir)
    data_root = workspace_dir / "data"
    checkpoint_dir = workspace_dir / "checkpoints"
    results_dir = Path(args.results_dir)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected DiffusionDB part ids: {part_ids}")

    manifest = download_subset(args, data_root, part_ids)
    train_records = manifest["train_records"]
    eval_records = manifest["eval_records"]
    print(
        f"Prepared {len(train_records)} training pairs and "
        f"{len(eval_records)} evaluation pairs."
    )

    text_model, _ = clip.load(args.clip_model, device=device, jit=False)
    text_model.eval()
    for parameter in text_model.parameters():
        parameter.requires_grad = False

    model, checkpoint_path = train_model(
        args=args,
        train_records=train_records,
        data_root=data_root,
        checkpoint_dir=checkpoint_dir,
        device=device,
        text_model=text_model,
    )

    prompt_cases = build_prompt_cases(eval_records, args.seed)
    case_results = {}
    case_summaries = {}
    for case_name in PROMPT_CASES:
        print(f"Evaluating prompt case: {case_name}")
        results = evaluate_case(
            args=args,
            case_name=case_name,
            prompts=prompt_cases[case_name],
            eval_records=eval_records,
            data_root=data_root,
            model=model,
            text_model=text_model,
            device=device,
        )
        case_results[case_name] = results
        case_summaries[case_name] = summarize_case(results)
        print(
            f"{case_name}: avg_image_bitstream={case_summaries[case_name]['average_image_bitstream']:.1f} "
            f"avg_compressed_size={case_summaries[case_name]['average_compressed_size']:.1f} "
            f"avg_bpp={case_summaries[case_name]['average_bpp']:.4f}"
        )

    save_results(
        args=args,
        results_dir=results_dir,
        case_results=case_results,
        summary=case_summaries,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    print(f"Saved result arrays to {results_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
