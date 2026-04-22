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
from model.dataset import MyDataset

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
    "SD21B": {"folder": "SD-2_1-B", "eval_size": (512, 512)},
    "SD21": {"folder": "SD-2_1", "eval_size": (768, 768)},
    "SDXL": {"folder": "SD-XL", "eval_size": (1024, 1024)},
    "MJ": {"folder": "MJ", "eval_size": (1024, 1024)},
    "MOD": {"folder": "MOD", "eval_size": (1408, 640)},
}


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


def resolve_text_file(split_root: Path, split: str) -> Path:
    candidates = {
        "train": ["train.txt"],
        "val": ["vaild.txt", "valid.txt", "val.txt"],
        "test": ["test.txt"],
    }[split]
    for candidate in candidates:
        path = split_root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Unable to locate the text file for split '{split}' under {split_root}."
    )


def build_dataloaders(args, device: torch.device) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    train_loaders = {}
    val_loaders = {}
    pin_memory = device.type == "cuda"

    for dataset_name, meta in DATASET_LAYOUT.items():
        split_root = Path(args.dataset) / meta["folder"]
        train_dataset = MyDataset(
            split_root / "train",
            resolve_text_file(split_root, "train"),
            transforms.Compose(
                [
                    transforms.RandomCrop(tuple(args.patch_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
        val_dataset = MyDataset(
            split_root / "val",
            resolve_text_file(split_root, "val"),
            transforms.Compose(
                [
                    transforms.CenterCrop(meta["eval_size"]),
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
        val_loaders[dataset_name] = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    return train_loaders, val_loaders


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
) -> None:
    model.train()
    device = next(model.parameters()).device

    for step, (images, prompts) in enumerate(loader):
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

        if step % 100 == 0:
            if metric == "mse":
                print(
                    f"Train epoch {epoch}: [{step * len(images)}/{len(loader.dataset)} "
                    f"({100.0 * step / max(len(loader), 1):.0f}%)] "
                    f"loss={out_loss['loss'].item():.4f} "
                    f"mse={out_loss['mse_loss'].item():.6f} "
                    f"bpp={out_loss['bpp_loss'].item():.4f} "
                    f"aux={aux_loss.item():.4f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: [{step * len(images)}/{len(loader.dataset)} "
                    f"({100.0 * step / max(len(loader), 1):.0f}%)] "
                    f"loss={out_loss['loss'].item():.4f} "
                    f"ms-ssim={out_loss['ms_ssim_loss'].item():.6f} "
                    f"bpp={out_loss['bpp_loss'].item():.4f} "
                    f"aux={aux_loss.item():.4f}"
                )


def evaluate(
    epoch: int,
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
        f"Validation epoch {epoch}: "
        f"loss={loss_meter.avg:.4f} "
        f"{metric_name}={distortion_meter.avg:.4f} "
        f"bpp={bpp_meter.avg:.4f} "
        f"aux={aux_meter.avg:.4f}"
    )
    return loss_meter.avg


def save_checkpoint(state: dict, epoch: int, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / f"{epoch}.pth.tar")
    torch.save(state, save_dir / "checkpoint_latest.pth.tar")


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
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--clip-model", default="ViT-B/32", type=str)
    parser.add_argument("--hidden-channels", default=128, type=int)
    parser.add_argument("--latent-channels", default=320, type=int)
    parser.add_argument("--hyper-channels", default=192, type=int)
    parser.add_argument("--text-dim", default=512, type=int)
    parser.add_argument("--num-slices", default=5, type=int)
    parser.add_argument("--save", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv) -> None:
    args = parse_args(argv)

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
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_epoch, gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    writer = None
    save_dir = Path(args.save_path) / f"{args.hidden_channels}_{args.lmbda}"
    if SummaryWriter is not None:
        writer = SummaryWriter(str(save_dir) + "_tensorboard")

    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "lr_scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch} | lr={optimizer.param_groups[0]['lr']:.2e}")
        for dataset_name in TRAIN_SCHEDULE:
            train_one_loader(
                model=model,
                text_model=text_model,
                loader=train_loaders[dataset_name],
                criterion=criterion,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                metric=args.metric,
            )

        val_loss = evaluate(epoch, model, text_model, val_loaders, criterion, args.metric)
        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
        scheduler.step()

        best_loss = min(best_loss, val_loss)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": val_loss,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                epoch,
                save_dir,
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
