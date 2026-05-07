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
class PromptCase:
    key: str
    label: str
    suffix: str
    color: str
    marker: str


@dataclass(frozen=True)
class RDPoint:
    lmbda: float
    bpp: float
    psnr: float
    ms_ssim: float
    image_count: int
    source_path: Path


PROMPT_CASES = [
    PromptCase("correct", "Correct prompts", "", "#0072B2", "o"),
    PromptCase("empty", "Empty prompts", "_empty_prompts", "#D55E00", "s"),
    PromptCase("shuffled", "Word-shuffled prompts", "_word_shuffled_prompts", "#009E73", "^"),
    PromptCase("swapped", "Swapped prompts", "_swapped_prompts", "#CC79A7", "x"),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MJ prompt-ablation R-D curves from CATC evaluator outputs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing mj_<lambda>*.txt result files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output figure path. Defaults to <results-dir>/mj_prompt_ablation_rd_curve.png.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        default=DEFAULT_LAMBDAS,
        type=float,
        help="Lambda values to look for and plot.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.key for case in PROMPT_CASES],
        default=[case.key for case in PROMPT_CASES],
        help="Prompt cases to plot.",
    )
    parser.add_argument("--dpi", default=200, type=int, help="Output figure DPI.")
    parser.add_argument(
        "--annotate-lambdas",
        action="store_true",
        help="Annotate each plotted point with its lambda value.",
    )
    return parser.parse_args(argv)


def lambda_to_tag(lmbda: float) -> str:
    return f"{lmbda:g}".replace(".", "_")


def result_path(results_dir: Path, lmbda: float, prompt_case: PromptCase) -> Path:
    return results_dir / f"mj_{lambda_to_tag(lmbda)}{prompt_case.suffix}_results.txt"


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

    image_count = len(image_rows)
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
        print(f"Skipping lambda={lmbda:g}: no metrics found in {path}")
        return None

    return RDPoint(
        lmbda=lmbda,
        bpp=sum(row["bpp"] for row in image_rows) / image_count,
        psnr=sum(row["psnr"] for row in image_rows) / image_count,
        ms_ssim=sum(row["ms_ssim"] for row in image_rows) / image_count,
        image_count=image_count,
        source_path=path,
    )


def load_case_points(
    results_dir: Path, lambdas: list[float], prompt_case: PromptCase
) -> list[RDPoint]:
    points = []
    for lmbda in lambdas:
        path = result_path(results_dir, lmbda, prompt_case)
        if not path.is_file():
            print(f"Skipping {prompt_case.label}, lambda={lmbda:g}: missing {path}")
            continue

        point = parse_result_file(path, lmbda)
        if point is None:
            continue

        points.append(point)
        print(
            f"Loaded {prompt_case.label}, lambda={point.lmbda:g}: "
            f"bpp={point.bpp:.4f}, psnr={point.psnr:.2f} dB, "
            f"ms-ssim={point.ms_ssim:.4f} dB, images={point.image_count}"
        )
    return sorted(points, key=lambda point: point.bpp)


def annotate_points(ax, points: list[RDPoint], values: list[float]) -> None:
    for point, y_value in zip(points, values):
        ax.annotate(
            f"{point.lmbda:g}",
            (point.bpp, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def padded_limits(values: list[float], min_pad: float) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    span = upper - lower
    pad = max(span * 0.08, min_pad)
    return lower - pad, upper + pad


def plot_curves(
    case_to_points: dict[PromptCase, list[RDPoint]],
    output_path: Path,
    dpi: int,
    annotate_lambdas: bool,
) -> None:
    if not any(case_to_points.values()):
        raise ValueError("No result points were available to plot.")

    all_points = [point for points in case_to_points.values() for point in points]
    xlim = padded_limits([point.bpp for point in all_points], min_pad=0.01)
    psnr_ylim = padded_limits([point.psnr for point in all_points], min_pad=0.03)
    ms_ssim_ylim = padded_limits([point.ms_ssim for point in all_points], min_pad=0.03)

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 7.8), constrained_layout=True)
    panels = [
        (axes[0], "psnr", "PSNR [dB]", psnr_ylim),
        (axes[1], "ms_ssim", "MS-SSIM [dB]", ms_ssim_ylim),
    ]

    for ax, metric, ylabel, ylim in panels:
        for prompt_case, points in case_to_points.items():
            if not points:
                continue

            bpp = [point.bpp for point in points]
            values = [getattr(point, metric) for point in points]
            ax.plot(
                bpp,
                values,
                marker=prompt_case.marker,
                markersize=5.5,
                linewidth=1.35,
                color=prompt_case.color,
                label=prompt_case.label,
            )
            if annotate_lambdas:
                annotate_points(ax, points, values)

        ax.set_xlabel("Bit-rate [bpp]")
        ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#b0b0b0", alpha=0.55, linewidth=0.8)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("MJ Prompt-Ablation R-D Curves")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    output_path = args.output or args.results_dir / "mj_prompt_ablation_rd_curve.png"
    selected_cases = [case for case in PROMPT_CASES if case.key in args.cases]
    case_to_points = {
        prompt_case: load_case_points(args.results_dir, args.lambdas, prompt_case)
        for prompt_case in selected_cases
    }
    plot_curves(case_to_points, output_path, args.dpi, args.annotate_lambdas)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
