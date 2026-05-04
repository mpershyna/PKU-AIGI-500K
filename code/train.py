from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms

from model.cm_gru import CM_GRU
from model.dataset import IMAGE_EXTENSIONS, MyDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional during setup
    SummaryWriter = None

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_SCHEDULE = [
    "MOD",
    "SD21B",
    "MOD",
    "MJ",
    "MOD",
    "SD21",
    "MOD",
    "MJ",
    "MOD",
    "SDXL",
    "MOD",
    "MJ",
    "MOD",
    "MJ",
]

DATASET_LAYOUT = {
    "SD21B": {
        "folder": "SD-2_1-B",
        "archive_folder": "SD21B",
        "eval_size": (512, 512),
        "train_text": ["train.txt", "SD-2_1-B.txt", "SD21B.txt"],
        "val_text": ["vaild.txt", "valid.txt", "val.txt", "SD-2_1-B.txt", "SD21B.txt"],
    },
    "SD21": {
        "folder": "SD-2_1",
        "archive_folder": "SD21",
        "eval_size": (768, 768),
        "train_text": ["train.txt", "SD21.txt", "SD-2_1.txt"],
        "val_text": ["vaild.txt", "valid.txt", "val.txt", "SD-2_1.txt", "SD21.txt"],
    },
    "SDXL": {
        "folder": "SD-XL",
        "archive_folder": "SDXL",
        "eval_size": (1024, 1024),
        "train_text": ["train.txt", "SD-XL.txt", "SDXL.txt"],
        "val_text": ["vaild.txt", "valid.txt", "val.txt", "SD-XL.txt", "SDXL.txt"],
    },
    "MJ": {
        "folder": "MJ",
        "archive_folder": "MJ",
        "eval_size": (1024, 1024),
        "train_text": ["train.txt", "MJ.txt"],
        "val_text": ["vaild.txt", "valid.txt", "val.txt", "MJ.txt"],
    },
    "MOD": {
        "folder": "MOD",
        "archive_folder": "MOD",
        "eval_size": (1408, 640),
        "train_text": ["train.txt", "MOD.txt"],
        "val_text": ["vaild.txt", "valid.txt", "val.txt", "MOD.txt"],
    },
}

DEFAULT_STEP_MILESTONES = [1_600_000, 1_850_000]


def compute_msssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ms_ssim(a, b, data_range=1.0)


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda: float = 1e-2, metric: str = "mse") -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = target.size()
        num_pixels = batch * height * width
        bpp_loss = sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        out = {"bpp_loss": bpp_loss}
        if self.metric == "mse":
            mse_loss = self.mse(output["x_hat"], target)
            out["mse_loss"] = mse_loss
            out["loss"] = self.lmbda * 255**2 * mse_loss + bpp_loss
            return out

        ms_ssim_loss = compute_msssim(output["x_hat"], target)
        out["ms_ssim_loss"] = ms_ssim_loss
        out["loss"] = self.lmbda * (1.0 - ms_ssim_loss) + bpp_loss
        return out


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        value = float(val)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def configure_optimizers(net: nn.Module, args) -> tuple[optim.Optimizer, optim.Optimizer]:
    params = {
        name
        for name, param in net.named_parameters()
        if not name.endswith(".quantiles") and param.requires_grad
    }
    aux_params = {
        name
        for name, param in net.named_parameters()
        if name.endswith(".quantiles") and param.requires_grad
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[name] for name in sorted(params)), lr=args.learning_rate)
    aux_optimizer = optim.Adam(
        (params_dict[name] for name in sorted(aux_params)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def encode_text_batch(text_model, prompts, device: torch.device) -> torch.Tensor:
    tokens = clip.tokenize(list(prompts), truncate=True).to(device)
    with torch.no_grad():
        return text_model.encode_text(tokens).float()


def has_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in path.iterdir()
    )


def resolve_dataset_names(requested: list[str]) -> list[str]:
    seen = set()
    datasets = []
    for dataset_name in requested:
        if dataset_name not in DATASET_LAYOUT:
            valid = ", ".join(DATASET_LAYOUT)
            raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}.")
        if dataset_name not in seen:
            datasets.append(dataset_name)
            seen.add(dataset_name)
    return datasets


def build_train_schedule(selected_datasets: list[str]) -> list[str]:
    selected = set(selected_datasets)
    schedule = [dataset_name for dataset_name in TRAIN_SCHEDULE if dataset_name in selected]
    for dataset_name in selected_datasets:
        if dataset_name not in schedule:
            schedule.append(dataset_name)
    return schedule


def resolve_subset_root(dataset_root: Path, dataset_name: str, meta: dict) -> Path:
    candidates = [
        dataset_root / meta["folder"],
        dataset_root / meta["archive_folder"],
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate the {dataset_name} directory under {dataset_root}. "
        f"Searched: {', '.join(str(path) for path in candidates)}."
    )


def resolve_text_file(
    search_roots: list[Path],
    candidates: list[str],
    dataset_name: str,
    split: str,
) -> Path:
    checked = []
    for root in search_roots:
        for candidate in candidates:
            path = root / candidate
            checked.append(path)
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"Unable to locate the {split} prompt file for {dataset_name}. "
        f"Searched: {', '.join(str(path) for path in checked)}."
    )


def resolve_image_dir(
    candidates: list[Path],
    dataset_name: str,
    split: str,
) -> Path:
    checked = []
    for candidate in candidates:
        checked.append(candidate)
        if has_images(candidate):
            return candidate

    # Official archives may add one extra named directory inside the subset root.
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for child in sorted(candidate.iterdir()):
            checked.append(child)
            if has_images(child):
                return child

    raise FileNotFoundError(
        f"Unable to locate {split} images for {dataset_name}. "
        f"Searched: {', '.join(str(path) for path in checked)}."
    )


def build_dataloaders(args, device: torch.device) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    train_loaders = {}
    val_loaders = {}
    pin_memory = device.type == "cuda"
    dataset_root = Path(args.dataset)
    validation_root = Path(args.validation_root) if args.validation_root else dataset_root
    selected_datasets = resolve_dataset_names(args.datasets)

    for dataset_name in selected_datasets:
        meta = DATASET_LAYOUT[dataset_name]
        split_root = resolve_subset_root(dataset_root, dataset_name, meta)
        archive_root = split_root / meta["archive_folder"]
        train_image_dir = resolve_image_dir(
            [
                split_root / "train",
                split_root / "images",
                archive_root,
                split_root,
            ],
            dataset_name,
            "training",
        )
        train_text_path = resolve_text_file(
            [split_root, archive_root],
            meta["train_text"],
            dataset_name,
            "training",
        )
        train_dataset = MyDataset(
            train_image_dir,
            train_text_path,
            transforms.Compose(
                [
                    transforms.RandomCrop(tuple(args.patch_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
        train_loaders[dataset_name] = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        print(
            f"{dataset_name} training: {len(train_dataset)} images from {train_image_dir}; "
            f"prompts from {train_text_path}"
        )

        if args.no_validation:
            continue

        validation_archive_root = validation_root / "vaild"
        validation_valid_root = validation_root / "valid"
        val_image_dir = resolve_image_dir(
            [
                validation_archive_root / meta["archive_folder"],
                validation_valid_root / meta["archive_folder"],
                validation_root / meta["archive_folder"],
                split_root / "val",
                split_root / "vaild",
                split_root / "valid",
            ],
            dataset_name,
            "validation",
        )
        val_text_path = resolve_text_file(
            [
                validation_archive_root,
                validation_valid_root,
                validation_root,
                split_root,
                split_root / "val",
                split_root / "vaild",
                split_root / "valid",
            ],
            meta["val_text"],
            dataset_name,
            "validation",
        )
        val_dataset = MyDataset(
            val_image_dir,
            val_text_path,
            transforms.Compose(
                [
                    transforms.CenterCrop(meta["eval_size"]),
                    transforms.ToTensor(),
                ]
            ),
        )
        val_loaders[dataset_name] = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        print(
            f"{dataset_name} validation: {len(val_dataset)} images from {val_image_dir}; "
            f"prompts from {val_text_path}"
        )

    return train_loaders, val_loaders


def train_batch(
    model: nn.Module,
    text_model,
    batch: tuple[torch.Tensor, tuple[str, ...]],
    criterion: RateDistortionLoss,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    clip_max_norm: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    device = next(model.parameters()).device
    images, prompts = batch
    images = images.to(device, non_blocking=True)
    text_features = encode_text_batch(text_model, prompts, device)

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    out_net = model(images, text_features)
    out_loss = criterion(out_net, images)
    out_loss["loss"].backward()

    if clip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()

    aux_loss = model.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()
    return out_loss, aux_loss


def print_train_metrics(
    prefix: str,
    out_loss: dict[str, torch.Tensor],
    aux_loss: torch.Tensor,
    metric: str,
) -> None:
    if metric == "mse":
        print(
            f"{prefix} "
            f"loss={out_loss['loss'].item():.4f} "
            f"mse={out_loss['mse_loss'].item():.6f} "
            f"bpp={out_loss['bpp_loss'].item():.4f} "
            f"aux={aux_loss.item():.4f}"
        )
    else:
        print(
            f"{prefix} "
            f"loss={out_loss['loss'].item():.4f} "
            f"ms-ssim={out_loss['ms_ssim_loss'].item():.6f} "
            f"bpp={out_loss['bpp_loss'].item():.4f} "
            f"aux={aux_loss.item():.4f}"
        )


def train_one_loader(
    model: nn.Module,
    text_model,
    loader: DataLoader,
    criterion: RateDistortionLoss,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    epoch: int,
    clip_max_norm: float,
    metric: str,
    log_interval: int = 100,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for step, batch in enumerate(loader):
        batch_size = batch[0].size(0)
        out_loss, aux_loss = train_batch(
            model=model,
            text_model=text_model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            clip_max_norm=clip_max_norm,
        )
        loss_meter.update(out_loss["loss"], batch_size)

        if log_interval > 0 and step % log_interval == 0:
            prefix = (
                f"Train epoch {epoch}: [{step * batch_size}/{len(loader.dataset)} "
                f"({100.0 * step / max(len(loader), 1):.0f}%)]"
            )
            print_train_metrics(prefix, out_loss, aux_loss, metric)

    return loss_meter.avg


def next_training_batch(
    dataset_name: str,
    train_loaders: dict[str, DataLoader],
    loader_iterators: dict[str, object],
):
    try:
        return next(loader_iterators[dataset_name])
    except StopIteration:
        loader_iterators[dataset_name] = iter(train_loaders[dataset_name])
        return next(loader_iterators[dataset_name])


def maybe_validate(
    progress_label: str,
    progress_value: int,
    model: nn.Module,
    text_model,
    val_loaders: dict[str, DataLoader],
    criterion: RateDistortionLoss,
    metric: str,
    fallback_loss: float,
) -> float:
    if val_loaders:
        return evaluate(progress_label, progress_value, model, text_model, val_loaders, criterion, metric)
    print(
        f"Validation skipped at {progress_label} {progress_value}: "
        "no validation loaders were configured."
    )
    return fallback_loss


def evaluate(
    progress_label: str,
    progress_value: int,
    model: nn.Module,
    text_model,
    val_loaders: dict[str, DataLoader],
    criterion: RateDistortionLoss,
    metric: str,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    distortion_meter = AverageMeter()
    aux_meter = AverageMeter()

    with torch.no_grad():
        for loader in val_loaders.values():
            for images, prompts in loader:
                images = images.to(device, non_blocking=True)
                text_features = encode_text_batch(text_model, prompts, device)
                out_net = model(images, text_features)
                out_loss = criterion(out_net, images)

                aux_meter.update(model.aux_loss())
                bpp_meter.update(out_loss["bpp_loss"])
                loss_meter.update(out_loss["loss"])
                if metric == "mse":
                    psnr = -10.0 * torch.log10(out_loss["mse_loss"])
                    distortion_meter.update(psnr)
                else:
                    distortion_meter.update(out_loss["ms_ssim_loss"])

    metric_name = "psnr" if metric == "mse" else "ms-ssim"
    print(
        f"Validation {progress_label} {progress_value}: "
        f"loss={loss_meter.avg:.4f} "
        f"{metric_name}={distortion_meter.avg:.4f} "
        f"bpp={bpp_meter.avg:.4f} "
        f"aux={aux_meter.avg:.4f}"
    )
    return loss_meter.avg


def build_checkpoint_payload(
    model: nn.Module,
    epoch: int,
    global_step: int,
    loss: float,
    best_loss: float,
    args,
    include_training_state: bool,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> dict:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unwrap_model(model).state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "args": vars(args),
        "checkpoint_mode": "full" if include_training_state else "weights_only",
    }
    if include_training_state:
        checkpoint.update(
            {
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
            }
        )
    return checkpoint


def save_checkpoint(
    state: dict,
    marker: int,
    save_dir: Path,
    save_epoch_checkpoints: bool = False,
    marker_name: str = "epoch",
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / "checkpoint_latest.pth.tar")
    if save_epoch_checkpoints:
        if marker_name == "epoch":
            filename = f"{marker}.pth.tar"
        else:
            filename = f"{marker_name}_{marker}.pth.tar"
        torch.save(state, save_dir / filename)


def run_epoch_training(
    args,
    model: nn.Module,
    text_model,
    train_loaders: dict[str, DataLoader],
    val_loaders: dict[str, DataLoader],
    criterion: RateDistortionLoss,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    save_dir: Path,
    writer,
    start_epoch: int,
    start_global_step: int,
    best_loss: float,
) -> None:
    train_schedule = build_train_schedule(resolve_dataset_names(args.datasets))
    global_step = start_global_step

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch} | lr={optimizer.param_groups[0]['lr']:.2e}")
        last_train_loss = float("inf")
        for dataset_name in train_schedule:
            print(f"Training subset: {dataset_name}")
            train_loss = train_one_loader(
                model=model,
                text_model=text_model,
                loader=train_loaders[dataset_name],
                criterion=criterion,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                metric=args.metric,
                log_interval=args.log_interval_steps,
            )
            global_step += len(train_loaders[dataset_name])
            last_train_loss = train_loss
            if writer is not None:
                writer.add_scalar(f"train/{dataset_name}_loss", train_loss, global_step)

        fallback_loss = last_train_loss if math.isfinite(last_train_loss) else best_loss
        val_loss = maybe_validate(
            "epoch",
            epoch,
            model,
            text_model,
            val_loaders,
            criterion,
            args.metric,
            fallback_loss,
        )
        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
        scheduler.step()

        best_loss = min(best_loss, val_loss)
        if args.save:
            checkpoint = build_checkpoint_payload(
                model=model,
                epoch=epoch,
                global_step=global_step,
                loss=val_loss,
                best_loss=best_loss,
                args=args,
                include_training_state=args.save_training_state,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
                scheduler=scheduler,
            )
            save_checkpoint(
                checkpoint,
                epoch,
                save_dir,
                save_epoch_checkpoints=args.save_epoch_checkpoints,
            )


def run_step_training(
    args,
    model: nn.Module,
    text_model,
    train_loaders: dict[str, DataLoader],
    val_loaders: dict[str, DataLoader],
    criterion: RateDistortionLoss,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    save_dir: Path,
    writer,
    start_global_step: int,
    best_loss: float,
) -> None:
    train_schedule = build_train_schedule(resolve_dataset_names(args.datasets))
    loader_iterators = {
        dataset_name: iter(train_loaders[dataset_name]) for dataset_name in train_loaders
    }
    global_step = start_global_step
    last_loss = best_loss if math.isfinite(best_loss) else float("inf")
    last_eval_step = -1

    if global_step >= args.max_steps:
        print(
            f"Checkpoint global_step={global_step} is already at or beyond "
            f"--max-steps={args.max_steps}; skipping training."
        )

    while global_step < args.max_steps:
        dataset_name = train_schedule[global_step % len(train_schedule)]
        batch = next_training_batch(dataset_name, train_loaders, loader_iterators)
        model.train()
        out_loss, aux_loss = train_batch(
            model=model,
            text_model=text_model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            clip_max_norm=args.clip_max_norm,
        )
        global_step += 1
        scheduler.step()
        last_loss = float(out_loss["loss"].detach())

        if writer is not None:
            writer.add_scalar("train/loss", last_loss, global_step)
            writer.add_scalar("train/bpp", float(out_loss["bpp_loss"].detach()), global_step)
            writer.add_scalar("train/aux_loss", float(aux_loss.detach()), global_step)

        if args.log_interval_steps > 0 and (
            global_step == 1 or global_step % args.log_interval_steps == 0
        ):
            prefix = (
                f"Train step {global_step}/{args.max_steps} "
                f"| subset={dataset_name} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            print_train_metrics(prefix, out_loss, aux_loss, args.metric)

        checkpoint_loss = last_loss
        if (
            args.validation_interval_steps > 0
            and global_step % args.validation_interval_steps == 0
        ):
            checkpoint_loss = maybe_validate(
                "step",
                global_step,
                model,
                text_model,
                val_loaders,
                criterion,
                args.metric,
                last_loss,
            )
            last_eval_step = global_step
            best_loss = min(best_loss, checkpoint_loss)
            if writer is not None:
                writer.add_scalar("val/loss", checkpoint_loss, global_step)

        if (
            args.save
            and args.checkpoint_interval_steps > 0
            and global_step % args.checkpoint_interval_steps == 0
        ):
            checkpoint = build_checkpoint_payload(
                model=model,
                epoch=-1,
                global_step=global_step,
                loss=checkpoint_loss,
                best_loss=best_loss,
                args=args,
                include_training_state=args.save_training_state,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
                scheduler=scheduler,
            )
            save_checkpoint(
                checkpoint,
                global_step,
                save_dir,
                save_epoch_checkpoints=args.save_epoch_checkpoints,
                marker_name="step",
            )

    if args.validation_interval_steps > 0 and val_loaders and last_eval_step != global_step:
        final_loss = evaluate(
            "step",
            global_step,
            model,
            text_model,
            val_loaders,
            criterion,
            args.metric,
        )
        best_loss = min(best_loss, final_loss)
        if writer is not None:
            writer.add_scalar("val/loss", final_loss, global_step)
    else:
        final_loss = last_loss if math.isfinite(last_loss) else best_loss

    if args.save:
        checkpoint = build_checkpoint_payload(
            model=model,
            epoch=-1,
            global_step=global_step,
            loss=final_loss,
            best_loss=best_loss,
            args=args,
            include_training_state=args.save_training_state,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            scheduler=scheduler,
        )
        save_checkpoint(
            checkpoint,
            global_step,
            save_dir,
            save_epoch_checkpoints=args.save_epoch_checkpoints,
            marker_name="step",
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train CATC on PKU-AIGI-500K.")
    parser.add_argument("-d", "--dataset", required=True, type=str, help="Dataset root")
    parser.add_argument("--save-path", required=True, type=str, help="Directory for checkpoints")
    parser.add_argument(
        "-m",
        "--model",
        default="CM_GRU",
        choices=["CM_GRU"],
        help="Model name kept for compatibility with the original release.",
    )
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument(
        "--training-mode",
        choices=["epoch", "step"],
        default="epoch",
        help=(
            "Use the original epoch loop or the paper-style optimizer-step schedule. "
            "Step mode defaults to 2,000,000 updates if --max-steps is not set."
        ),
    )
    parser.add_argument(
        "--max-steps",
        default=None,
        type=int,
        help="Number of optimizer updates to run in step mode.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LAYOUT),
        choices=list(DATASET_LAYOUT),
        help="Subset names to include. Defaults to all PKU-AIGI-500K training subsets.",
    )
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--test-batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--aux-learning-rate", default=5e-4, type=float)
    parser.add_argument("--lambda", dest="lmbda", default=0.05, type=float)
    parser.add_argument("--patch-size", nargs=2, default=(256, 256), type=int)
    parser.add_argument("--clip-max-norm", default=1.0, type=float)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--metric", default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--lr-epoch", nargs="*", default=[45, 48], type=int)
    parser.add_argument(
        "--lr-step",
        nargs="*",
        default=DEFAULT_STEP_MILESTONES,
        type=int,
        help=(
            "Optimizer-step milestones for step mode. The CATC paper describes "
            "drops after 1.6M and 1.85M steps."
        ),
    )
    parser.add_argument(
        "--log-interval-steps",
        default=100,
        type=int,
        help="Training log interval, measured in optimizer updates.",
    )
    parser.add_argument(
        "--validation-interval-steps",
        default=50000,
        type=int,
        help="Validation interval in step mode. Set to 0 to skip periodic validation.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        default=50000,
        type=int,
        help="Checkpoint interval in step mode. Set to 0 to save only at the end.",
    )
    parser.add_argument(
        "--validation-root",
        default=None,
        type=str,
        help=(
            "Optional root containing the extracted validation archive. "
            "If omitted, the dataset root is searched."
        ),
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Train without constructing validation loaders.",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--clip-model", default="ViT-B/32", type=str)
    parser.add_argument("--hidden-channels", default=128, type=int)
    parser.add_argument("--latent-channels", default=320, type=int)
    parser.add_argument("--hyper-channels", default=192, type=int)
    parser.add_argument("--text-dim", default=512, type=int)
    parser.add_argument("--num-slices", default=5, type=int)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument(
        "--save-training-state",
        action="store_true",
        help=(
            "Include optimizer, auxiliary optimizer, and scheduler state in checkpoints. "
            "This enables exact training resume but makes checkpoints much larger."
        ),
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help=(
            "Keep per-epoch checkpoint snapshots in addition to checkpoint_latest.pth.tar. "
            "By default only the latest checkpoint is retained."
        ),
    )
    return parser.parse_args(argv)


def main(argv) -> None:
    args = parse_args(argv)
    args.datasets = resolve_dataset_names(args.datasets)
    if args.training_mode == "step" and args.max_steps is None:
        args.max_steps = 2_000_000

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loaders, val_loaders = build_dataloaders(args, device)

    text_model, _ = clip.load(args.clip_model, device=device, jit=False)
    text_model.eval()
    for parameter in text_model.parameters():
        parameter.requires_grad = False

    model = CM_GRU(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        text_dim=args.text_dim,
        num_slices=args.num_slices,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)

    optimizer, aux_optimizer = configure_optimizers(model, args)
    lr_milestones = args.lr_step if args.training_mode == "step" else args.lr_epoch
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    writer = None
    save_dir = Path(args.save_path) / f"{args.hidden_channels}_{args.lmbda}"
    if SummaryWriter is not None:
        writer = SummaryWriter(str(save_dir) + "_tensorboard")

    start_epoch = 0
    start_global_step = 0
    best_loss = float("inf")
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Checkpoint does not include optimizer state; using a fresh optimizer.")
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        else:
            print("Checkpoint does not include auxiliary optimizer state; using a fresh one.")
        if "lr_scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            print("Checkpoint does not include scheduler state; using a fresh scheduler.")
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        start_global_step = int(checkpoint.get("global_step", 0))

    if args.training_mode == "epoch":
        run_epoch_training(
            args=args,
            model=model,
            text_model=text_model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            criterion=criterion,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            writer=writer,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            best_loss=best_loss,
        )
    else:
        run_step_training(
            args=args,
            model=model,
            text_model=text_model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            criterion=criterion,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            writer=writer,
            start_global_step=start_global_step,
            best_loss=best_loss,
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
