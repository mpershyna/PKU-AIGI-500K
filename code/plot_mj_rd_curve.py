from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LAMBDAS = [0.0083, 0.015, 0.0275, 0.05]
PAPER_PSNR_YLIM = (29.0, 39.2)
PAPER_MS_SSIM_YLIM = (14.2, 22.2)
SUMMARY_PATTERNS = {
    "psnr": re.compile(r"^average_PSNR:\s*([0-9.+-eE]+)dB\s*$"),
    "ms_ssim": re.compile(r"^average_MS-SSIM:\s*([0-9.+-eE]+)dB\s*$"),
    "bpp": re.compile(r"^average_Bit-rate:\s*([0-9.+-eE]+)\s*bpp\s*$"),
}
PER_IMAGE_PATTERN = re.compile(
    r"^(?P<image>.+?): .*?\bbpp=(?P<bpp>[0-9.+-eE]+) "
    r"psnr=(?P<psnr>[0-9.+-eE]+)dB "
    r"ms-ssim=(?P<ms_ssim>[0-9.+-eE]+)dB"
)


@dataclass(frozen=True)
class RDPoint:
    lmbda: float
    bpp: float
    psnr: float
    ms_ssim: float
    image_count: int
    source_path: Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MJ rate-distortion points from CATC evaluator text outputs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing mj_<lambda>_results.txt files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output figure path. Defaults to <results-dir>/mj_rd_curve.png.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        default=DEFAULT_LAMBDAS,
        type=float,
        help="Lambda values to look for and plot.",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--annotate-lambdas",
        action="store_true",
        help="Annotate each plotted point with its lambda value.",
    )
    return parser.parse_args(argv)


def lambda_to_tag(lmbda: float) -> str:
    return f"{lmbda:g}".replace(".", "_")


def result_path(results_dir: Path, lmbda: float) -> Path:
    return results_dir / f"mj_{lambda_to_tag(lmbda)}_results.txt"


def parse_result_file(path: Path, lmbda: float) -> RDPoint | None:
    image_rows = []
    summary = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            image_match = PER_IMAGE_PATTERN.match(stripped)
            if image_match is not None:
                image_rows.append(
                    {
                        "bpp": float(image_match.group("bpp")),
                        "psnr": float(image_match.group("psnr")),
                        "ms_ssim": float(image_match.group("ms_ssim")),
                    }
                )
                continue

            for key, pattern in SUMMARY_PATTERNS.items():
                summary_match = pattern.match(stripped)
                if summary_match is not None:
                    summary[key] = float(summary_match.group(1))
                    break

    if image_rows:
        image_count = len(image_rows)
    else:
        image_count = 0

    if {"bpp", "psnr", "ms_ssim"}.issubset(summary):
        return RDPoint(
            lmbda=lmbda,
            bpp=summary["bpp"],
            psnr=summary["psnr"],
            ms_ssim=summary["ms_ssim"],
            image_count=image_count,
            source_path=path,
        )

    if not image_rows:
        print(f"Skipping lambda={lmbda:g}: no per-image metrics found in {path}")
        return None

    return RDPoint(
        lmbda=lmbda,
        bpp=sum(row["bpp"] for row in image_rows) / image_count,
        psnr=sum(row["psnr"] for row in image_rows) / image_count,
        ms_ssim=sum(row["ms_ssim"] for row in image_rows) / image_count,
        image_count=image_count,
        source_path=path,
    )


def load_points(results_dir: Path, lambdas: list[float]) -> list[RDPoint]:
    points = []
    for lmbda in lambdas:
        path = result_path(results_dir, lmbda)
        if not path.is_file():
            print(f"Skipping lambda={lmbda:g}: missing {path}")
            continue
        point = parse_result_file(path, lmbda)
        if point is not None:
            points.append(point)
            print(
                f"Loaded lambda={point.lmbda:g}: "
                f"bpp={point.bpp:.4f}, psnr={point.psnr:.2f} dB, "
                f"ms-ssim={point.ms_ssim:.4f} dB, images={point.image_count}"
            )
    return sorted(points, key=lambda point: point.bpp)


def plot_points(points: list[RDPoint], output_path: Path, dpi: int, annotate_lambdas: bool) -> None:
    if not points:
        raise ValueError("No result points were available to plot.")

    bpp = [point.bpp for point in points]
    psnr = [point.psnr for point in points]
    ms_ssim = [point.ms_ssim for point in points]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
    color = "#ff66c4"

    panels = [
        (axes[0], psnr, "PSNR [dB]", PAPER_PSNR_YLIM),
        (axes[1], ms_ssim, "MS-SSIM [dB]", PAPER_MS_SSIM_YLIM),
    ]
    for ax, values, ylabel, ylim in panels:
        ax.plot(
            bpp,
            values,
            marker="+",
            markersize=9,
            linewidth=1.4,
            color=color,
            label="My implementation",
        )
        if annotate_lambdas:
            for point, y_value in zip(points, values):
                ax.annotate(
                    f"{point.lmbda:g}",
                    (point.bpp, y_value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_xlabel("Bit-rate [bpp]")
        ax.set_xlim(0.1, 0.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#b0b0b0", alpha=0.55, linewidth=0.8)
        ax.legend(loc="best")

    fig.suptitle("R-D curves tested on MJ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    output_path = args.output or args.results_dir / "mj_rd_curve.png"
    points = load_points(args.results_dir, args.lambdas)
    plot_points(points, output_path, args.dpi, args.annotate_lambdas)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
