from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_LAMBDAS = [0.0083, 0.015, 0.0275, 0.05]
METRICS = ["psnr", "ms_ssim", "bpp"]
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


@dataclass
class TestResult:
    prompt_case: PromptCase
    scope: str
    lmbda: float | None
    metric: str
    n: int
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float | None
    significant: bool | None
    correct_path: Path | None
    ablation_path: Path | None


PROMPT_CASES = [
    PromptCase("empty", "Empty prompts", "_empty_prompts"),
    PromptCase("shuffled", "Word-shuffled prompts", "_word_shuffled_prompts"),
    PromptCase("swapped", "Swapped prompts", "_swapped_prompts"),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether MJ prompt-ablation R-D metrics differ significantly "
            "from the correct-prompt case."
        )
    )
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--lambdas", nargs="+", default=DEFAULT_LAMBDAS, type=float)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.key for case in PROMPT_CASES],
        default=[case.key for case in PROMPT_CASES],
    )
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--permutations", default=10000, type=int)
    parser.add_argument("--bootstrap-samples", default=5000, type=int)
    parser.add_argument("--seed", default=500000, type=int)
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Text report path. Defaults to <results-dir>/mj_prompt_ablation_significance.txt.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        type=Path,
        help="CSV report path. Defaults to <results-dir>/mj_prompt_ablation_significance.csv.",
    )
    return parser.parse_args(argv)


def lambda_to_tag(lmbda: float) -> str:
    return f"{lmbda:g}".replace(".", "_")


def result_path(results_dir: Path, lmbda: float, suffix: str) -> Path:
    return results_dir / f"mj_{lambda_to_tag(lmbda)}{suffix}_results.txt"


def parse_result_file(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = PER_IMAGE_PATTERN.match(line.strip())
            if match is None:
                continue

            rows[match.group("image")] = {
                "bpp": float(match.group("bpp")),
                "psnr": float(match.group("psnr")),
                "ms_ssim": float(match.group("ms_ssim")),
            }
    return rows


def paired_differences(
    correct_rows: dict[str, dict[str, float]],
    ablation_rows: dict[str, dict[str, float]],
    metric: str,
) -> np.ndarray:
    images = sorted(correct_rows.keys() & ablation_rows.keys())
    if not images:
        return np.array([], dtype=np.float64)
    return np.array(
        [ablation_rows[image][metric] - correct_rows[image][metric] for image in images],
        dtype=np.float64,
    )


def sign_flip_p_value(
    diffs: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
    batch_size: int = 2048,
) -> float:
    if len(diffs) == 0:
        return float("nan")

    observed = abs(float(np.mean(diffs)))
    if observed == 0.0:
        return 1.0

    extreme = 0
    completed = 0
    while completed < permutations:
        batch = min(batch_size, permutations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(diffs)))
        permuted_means = np.mean(signs * diffs, axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted_means) >= observed))
        completed += batch

    return (extreme + 1.0) / (permutations + 1.0)


def bootstrap_ci(
    diffs: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    batch_size: int = 2048,
) -> tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")

    means = np.empty(samples, dtype=np.float64)
    completed = 0
    while completed < samples:
        batch = min(batch_size, samples - completed)
        indices = rng.integers(0, len(diffs), size=(batch, len(diffs)))
        means[completed : completed + batch] = np.mean(diffs[indices], axis=1)
        completed += batch

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_diffs(
    diffs: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
    bootstrap_samples: int,
) -> tuple[float, float, float, float, float]:
    return (
        float(np.mean(diffs)),
        float(np.median(diffs)),
        *bootstrap_ci(diffs, rng, bootstrap_samples),
        sign_flip_p_value(diffs, rng, permutations),
    )


def holm_adjust(results: list[TestResult], alpha: float) -> None:
    valid = [(i, result.p_value) for i, result in enumerate(results) if np.isfinite(result.p_value)]
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    running_max = 0.0

    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, adjusted)
        adjusted_by_index[index] = running_max

    for index, result in enumerate(results):
        result.p_holm = adjusted_by_index.get(index, float("nan"))
        result.significant = bool(result.p_holm <= alpha) if np.isfinite(result.p_holm) else False


def metric_label(metric: str) -> str:
    return {
        "psnr": "PSNR [dB]",
        "ms_ssim": "MS-SSIM [dB]",
        "bpp": "Bit-rate [bpp]",
    }[metric]


def load_all_results(
    results_dir: Path,
    lambdas: list[float],
    cases: list[PromptCase],
) -> tuple[
    dict[float, tuple[Path, dict[str, dict[str, float]]]],
    dict[tuple[str, float], tuple[Path, dict[str, dict[str, float]]]],
]:
    correct = {}
    ablations = {}

    for lmbda in lambdas:
        path = result_path(results_dir, lmbda, "")
        if path.is_file():
            correct[lmbda] = (path, parse_result_file(path))
        else:
            print(f"Skipping lambda={lmbda:g}: missing correct-prompt file {path}")

        for prompt_case in cases:
            case_path = result_path(results_dir, lmbda, prompt_case.suffix)
            if case_path.is_file():
                ablations[(prompt_case.key, lmbda)] = (case_path, parse_result_file(case_path))
            else:
                print(
                    f"Skipping {prompt_case.label}, lambda={lmbda:g}: "
                    f"missing {case_path}"
                )

    return correct, ablations


def curve_level_diffs(
    correct: dict[float, tuple[Path, dict[str, dict[str, float]]]],
    ablations: dict[tuple[str, float], tuple[Path, dict[str, dict[str, float]]]],
    prompt_case: PromptCase,
    lambdas: list[float],
    metric: str,
) -> np.ndarray:
    available_lambdas = [
        lmbda
        for lmbda in lambdas
        if lmbda in correct and (prompt_case.key, lmbda) in ablations
    ]
    if not available_lambdas:
        return np.array([], dtype=np.float64)

    common_images: set[str] | None = None
    for lmbda in available_lambdas:
        correct_rows = correct[lmbda][1]
        ablation_rows = ablations[(prompt_case.key, lmbda)][1]
        images = set(correct_rows.keys() & ablation_rows.keys())
        common_images = images if common_images is None else common_images & images

    if not common_images:
        return np.array([], dtype=np.float64)

    diffs = []
    for image in sorted(common_images):
        per_lambda = []
        for lmbda in available_lambdas:
            correct_rows = correct[lmbda][1]
            ablation_rows = ablations[(prompt_case.key, lmbda)][1]
            per_lambda.append(ablation_rows[image][metric] - correct_rows[image][metric])
        diffs.append(float(np.mean(per_lambda)))

    return np.array(diffs, dtype=np.float64)


def run_tests(args: argparse.Namespace) -> list[TestResult]:
    selected_cases = [case for case in PROMPT_CASES if case.key in args.cases]
    correct, ablations = load_all_results(args.results_dir, args.lambdas, selected_cases)
    rng = np.random.default_rng(args.seed)
    results: list[TestResult] = []

    for prompt_case in selected_cases:
        for metric in METRICS:
            diffs = curve_level_diffs(correct, ablations, prompt_case, args.lambdas, metric)
            if len(diffs) == 0:
                continue

            mean_diff, median_diff, ci_low, ci_high, p_value = summarize_diffs(
                diffs, rng, args.permutations, args.bootstrap_samples
            )
            results.append(
                TestResult(
                    prompt_case=prompt_case,
                    scope="curve",
                    lmbda=None,
                    metric=metric,
                    n=len(diffs),
                    mean_diff=mean_diff,
                    median_diff=median_diff,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_value,
                    p_holm=None,
                    significant=None,
                    correct_path=None,
                    ablation_path=None,
                )
            )

        for lmbda in args.lambdas:
            if lmbda not in correct or (prompt_case.key, lmbda) not in ablations:
                continue

            correct_path, correct_rows = correct[lmbda]
            ablation_path, ablation_rows = ablations[(prompt_case.key, lmbda)]
            for metric in METRICS:
                diffs = paired_differences(correct_rows, ablation_rows, metric)
                if len(diffs) == 0:
                    continue

                mean_diff, median_diff, ci_low, ci_high, p_value = summarize_diffs(
                    diffs, rng, args.permutations, args.bootstrap_samples
                )
                results.append(
                    TestResult(
                        prompt_case=prompt_case,
                        scope="lambda",
                        lmbda=lmbda,
                        metric=metric,
                        n=len(diffs),
                        mean_diff=mean_diff,
                        median_diff=median_diff,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        p_value=p_value,
                        p_holm=None,
                        significant=None,
                        correct_path=correct_path,
                        ablation_path=ablation_path,
                    )
                )

    holm_adjust(results, args.alpha)
    return results


def write_csv(path: Path, results: list[TestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "scope",
                "lambda",
                "metric",
                "n",
                "mean_diff_ablation_minus_correct",
                "median_diff",
                "ci95_low",
                "ci95_high",
                "p_value_sign_flip",
                "p_value_holm",
                "significant_holm",
                "correct_path",
                "ablation_path",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.prompt_case.key,
                    result.scope,
                    "" if result.lmbda is None else f"{result.lmbda:g}",
                    result.metric,
                    result.n,
                    f"{result.mean_diff:.10g}",
                    f"{result.median_diff:.10g}",
                    f"{result.ci_low:.10g}",
                    f"{result.ci_high:.10g}",
                    f"{result.p_value:.10g}",
                    f"{result.p_holm:.10g}" if result.p_holm is not None else "",
                    int(bool(result.significant)),
                    "" if result.correct_path is None else str(result.correct_path),
                    "" if result.ablation_path is None else str(result.ablation_path),
                ]
            )


def format_result_line(result: TestResult) -> str:
    lambda_label = "all" if result.lmbda is None else f"{result.lmbda:g}"
    significant = "yes" if result.significant else "no"
    return (
        f"{result.prompt_case.label:<22} "
        f"{result.scope:<6} "
        f"{lambda_label:<7} "
        f"{metric_label(result.metric):<14} "
        f"{result.n:>5d} "
        f"{result.mean_diff:>11.6f} "
        f"{result.median_diff:>11.6f} "
        f"[{result.ci_low:>10.6f}, {result.ci_high:>10.6f}] "
        f"{result.p_value:>10.5g} "
        f"{result.p_holm:>10.5g} "
        f"{significant:>4}"
    )


def write_text_report(path: Path, results: list[TestResult], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MJ prompt-ablation significance report",
        "",
        "Difference convention: ablation - correct.",
        "For PSNR and MS-SSIM, negative values mean the ablation is worse.",
        "For bit-rate, positive values mean the ablation uses more bits.",
        (
            "Curve-level rows average each image's paired difference across "
            "the available lambda values before testing."
        ),
        (
            "P-values use a paired sign-flip permutation test; confidence intervals "
            "are paired bootstrap intervals for the mean difference."
        ),
        (
            f"Holm correction is applied across all reported tests with alpha={args.alpha:g}. "
            f"permutations={args.permutations}, bootstrap_samples={args.bootstrap_samples}, "
            f"seed={args.seed}."
        ),
        "",
        (
            f"{'case':<22} {'scope':<6} {'lambda':<7} {'metric':<14} {'n':>5} "
            f"{'mean_diff':>11} {'median':>11} {'95% CI':>25} "
            f"{'p':>10} {'p_holm':>10} {'sig':>4}"
        ),
        "-" * 133,
    ]
    lines.extend(format_result_line(result) for result in results)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    text_output = args.output or args.results_dir / "mj_prompt_ablation_significance.txt"
    csv_output = args.csv_output or args.results_dir / "mj_prompt_ablation_significance.csv"

    results = run_tests(args)
    write_text_report(text_output, results, args)
    write_csv(csv_output, results)

    print(f"Wrote {text_output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
