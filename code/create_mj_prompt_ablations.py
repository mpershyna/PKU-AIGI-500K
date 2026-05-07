from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_prompts(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def infer_prompt_index(image_path: Path) -> int:
    match = re.match(r"(\d+)", image_path.stem)
    if match is None:
        raise ValueError(
            f"Unable to infer a prompt id from '{image_path.name}'. "
            "Expected the filename to start with digits."
        )
    return int(match.group(1)) - 1


def collect_test_prompt_indices(test_dir: Path, num_prompts: int) -> list[int]:
    image_paths = sorted(
        path for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    indices = []
    for image_path in image_paths:
        prompt_index = infer_prompt_index(image_path)
        if prompt_index < 0 or prompt_index >= num_prompts:
            raise IndexError(
                f"Prompt id {prompt_index + 1} from '{image_path.name}' is outside "
                f"the available prompt range [1, {num_prompts}]."
            )
        indices.append(prompt_index)
    return sorted(set(indices))


def shuffle_words(prompt: str, rng: random.Random) -> str:
    words = prompt.split()
    if len(words) <= 1:
        return prompt

    shuffled = words[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)


def make_derangement(values: list[int], rng: random.Random) -> list[int]:
    if len(values) < 2:
        raise ValueError("Need at least two prompt ids to create swapped prompts.")

    shuffled = values[:]
    for _ in range(10000):
        rng.shuffle(shuffled)
        if all(original != replacement for original, replacement in zip(values, shuffled)):
            return shuffled

    # Deterministic fallback: a cyclic shift is always a derangement for unique values.
    offset = rng.randrange(1, len(values))
    return values[offset:] + values[:offset]


def make_prompt_derangement(prompts: list[str], rng: random.Random) -> list[int]:
    indices = list(range(len(prompts)))
    if len(indices) < 2:
        raise ValueError("Need at least two prompts to create swapped prompts.")

    for _ in range(10000):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(i != j and prompts[i] != prompts[j] for i, j in zip(indices, shuffled)):
            return shuffled

    # Deterministic fallback: find a cyclic shift that avoids unchanged lines.
    for offset in range(1, len(indices)):
        shifted = indices[offset:] + indices[:offset]
        if all(i != j and prompts[i] != prompts[j] for i, j in zip(indices, shifted)):
            return shifted

    raise ValueError(
        "Unable to create a swapped-prompt file with every line changed. "
        "This can happen if too many prompt lines contain identical text."
    )


def create_prompt_files(args: argparse.Namespace) -> None:
    source_prompts = read_prompts(Path(args.source_prompts))
    test_indices = collect_test_prompt_indices(Path(args.test_dir), len(source_prompts))
    rng = random.Random(args.seed)

    empty_prompts = [""] * len(source_prompts)
    shuffled_word_prompts = [shuffle_words(prompt, rng) for prompt in source_prompts]

    swapped_indices = make_prompt_derangement(source_prompts, rng)
    swapped_prompts = [source_prompts[source_index] for source_index in swapped_indices]

    output_dir = Path(args.output_dir)
    empty_path = output_dir / args.empty_filename
    shuffled_path = output_dir / args.shuffled_filename
    swapped_path = output_dir / args.swapped_filename

    write_prompts(empty_path, empty_prompts)
    write_prompts(shuffled_path, shuffled_word_prompts)
    write_prompts(swapped_path, swapped_prompts)

    print(f"Source prompts: {args.source_prompts} ({len(source_prompts)} lines)")
    print(f"Test prompt ids: {len(test_indices)} unique ids from {args.test_dir}")
    print(f"Empty prompts: {empty_path}")
    print(f"Word-shuffled prompts: {shuffled_path}")
    print(f"Swapped prompts: {swapped_path}")
    print(f"Random seed: {args.seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MJ test prompt ablation files compatible with the repo's "
            "numeric-id prompt lookup."
        )
    )
    parser.add_argument("--source-prompts", default="data/MJ/MJ.txt", type=str)
    parser.add_argument("--test-dir", default="data/MJ/test", type=str)
    parser.add_argument("--output-dir", default="data/MJ", type=str)
    parser.add_argument("--seed", default=500000, type=int)
    parser.add_argument("--empty-filename", default="MJ_test_empty_prompts.txt", type=str)
    parser.add_argument(
        "--shuffled-filename", default="MJ_test_word_shuffled_prompts.txt", type=str
    )
    parser.add_argument("--swapped-filename", default="MJ_test_swapped_prompts.txt", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    create_prompt_files(parse_args())
