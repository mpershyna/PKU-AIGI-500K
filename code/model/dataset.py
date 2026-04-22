from __future__ import annotations

import os
import re
from pathlib import Path

from PIL import Image
from torch.utils.data.dataset import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MyDataset(Dataset):
    """Loads paired images and prompt text from PKU-AIGI-500K splits."""

    def __init__(self, image_dir: str, text_path: str, image_transform=None) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.text_path = Path(text_path)
        self.image_transform = image_transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.text_path.is_file():
            raise FileNotFoundError(f"Text file does not exist: {self.text_path}")

        with self.text_path.open("r", encoding="utf-8") as handle:
            self.text_list = [line.rstrip("\n") for line in handle]

        self.image_list = sorted(
            name
            for name in os.listdir(self.image_dir)
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_list)

    def _lookup_text(self, filename: str) -> str:
        match = re.match(r"(\d+)", Path(filename).stem)
        if match is None:
            raise ValueError(
                f"Unable to infer the prompt index from image name '{filename}'. "
                "Expected the file name to start with a numeric id."
            )

        prompt_index = int(match.group(1)) - 1
        if prompt_index < 0 or prompt_index >= len(self.text_list):
            raise IndexError(
                f"Prompt index {prompt_index + 1} derived from '{filename}' "
                f"is outside the available text range [1, {len(self.text_list)}]."
            )
        return self.text_list[prompt_index]

    def __getitem__(self, idx: int):
        filename = self.image_list[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        text = self._lookup_text(filename)
        return image, text
