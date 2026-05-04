from __future__ import annotations

import argparse
import bisect
import io
import os
import struct
import shutil
import sys
import time
import zipfile
from pathlib import Path


ARCHIVE_GROUPS = {
    "SD21": {
        "archives": [("SD-2_1/SD21.zip", "SD-2_1")],
    },
    "SD21B": {
        "archives": [
            ("SD-2_1-B/SD-2_1-B-part1.zip", "SD-2_1-B"),
            ("SD-2_1-B/SD-2_1-B-part2.zip", "SD-2_1-B"),
        ],
    },
    "SDXL": {
        "archives": [("SD-XL/SD-XL.zip", "SD-XL")],
    },
    "MJ": {
        "archives": [("MJ/MJ.zip", "MJ")],
    },
    "MOD": {
        "archives": [("MOD/MOD.zip", "MOD")],
    },
    "vaild": {
        "archives": [("vaild.zip", ".")],
    },
}


class SplitZipFile(io.RawIOBase):
    """Seekable virtual file over .z01, .z02, ..., .zip parts."""

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        self.parts = parts
        self.part_sizes = [part.stat().st_size for part in parts]
        self.part_offsets = [0]
        for size in self.part_sizes:
            self.part_offsets.append(self.part_offsets[-1] + size)
        self.total_size = self.part_offsets[-1]
        self.position = 0
        self._part_index: int | None = None
        self._part_handle = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.position + offset
        elif whence == os.SEEK_END:
            new_position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence value: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self.position = new_position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if self.position >= self.total_size:
            return b""
        if size is None or size < 0:
            size = self.total_size - self.position

        remaining = min(size, self.total_size - self.position)
        chunks = []
        while remaining > 0 and self.position < self.total_size:
            part_index = bisect.bisect_right(self.part_offsets, self.position) - 1
            part_index = min(part_index, len(self.parts) - 1)
            part_offset = self.position - self.part_offsets[part_index]
            bytes_from_part = min(
                remaining,
                self.part_sizes[part_index] - part_offset,
            )

            handle = self._open_part(part_index)
            handle.seek(part_offset)
            chunk = handle.read(bytes_from_part)
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._part_handle is not None:
            self._part_handle.close()
            self._part_handle = None
        super().close()

    def _open_part(self, part_index: int):
        if self._part_index != part_index:
            if self._part_handle is not None:
                self._part_handle.close()
            self._part_handle = self.parts[part_index].open("rb")
            self._part_index = part_index
        return self._part_handle


def split_zip_parts(archive_path: Path) -> list[Path]:
    z_parts = []
    for candidate in archive_path.parent.glob(f"{archive_path.stem}.z*"):
        suffix = candidate.suffix.lower()
        if len(suffix) == 4 and suffix.startswith(".z") and suffix[2:].isdigit():
            z_parts.append(candidate)
    z_parts.sort(key=lambda path: int(path.suffix[2:]))
    return z_parts + [archive_path]


def open_zip(archive_path: Path) -> tuple[zipfile.ZipFile, SplitZipFile]:
    parts = split_zip_parts(archive_path)
    missing = [part for part in parts if not part.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing archive parts for {archive_path}: "
            f"{', '.join(str(part) for part in missing)}"
        )

    reader = SplitZipFile(parts)
    end_record = find_end_record(reader)
    if end_record is None:
        reader.close()
        raise zipfile.BadZipFile(f"Could not find ZIP central directory: {archive_path}")

    original_end_record_parser = zipfile._EndRecData

    def patched_end_record_parser(file_handle):
        if file_handle is reader:
            return list(end_record)
        return original_end_record_parser(file_handle)

    zipfile._EndRecData = patched_end_record_parser
    try:
        zip_handle = zipfile.ZipFile(reader)
    finally:
        zipfile._EndRecData = original_end_record_parser

    patch_split_offsets(zip_handle, reader, end_record)
    return zip_handle, reader


def find_end_record(reader: SplitZipFile):
    reader.seek(0, os.SEEK_END)
    file_size = reader.tell()
    max_comment_start = max(file_size - (1 << 16) - zipfile.sizeEndCentDir, 0)
    reader.seek(max_comment_start)
    data = reader.read(file_size - max_comment_start)
    start = data.rfind(zipfile.stringEndArchive)
    if start < 0:
        return None

    record_data = data[start : start + zipfile.sizeEndCentDir]
    if len(record_data) != zipfile.sizeEndCentDir:
        return None
    end_record = list(struct.unpack(zipfile.structEndArchive, record_data))
    comment_size = end_record[zipfile._ECD_COMMENT_SIZE]
    comment = data[
        start + zipfile.sizeEndCentDir : start + zipfile.sizeEndCentDir + comment_size
    ]
    end_record.append(comment)
    end_record.append(max_comment_start + start)
    end_record = read_zip64_end_record(reader, max_comment_start + start, end_record)
    normalize_end_record_location(reader, end_record)
    return end_record


def read_zip64_end_record(
    reader: SplitZipFile,
    eocd_location: int,
    end_record: list,
) -> list:
    locator_offset = eocd_location - zipfile.sizeEndCentDir64Locator
    if locator_offset < 0:
        return end_record

    reader.seek(locator_offset)
    data = reader.read(zipfile.sizeEndCentDir64Locator)
    if len(data) != zipfile.sizeEndCentDir64Locator:
        raise OSError("Unable to read ZIP64 end-of-central-directory locator")
    signature, disk_number, relative_offset, disk_count = struct.unpack(
        zipfile.structEndArchive64Locator,
        data,
    )
    if signature != zipfile.stringEndArchive64Locator:
        return end_record
    if disk_number >= len(reader.parts):
        raise zipfile.BadZipFile(
            f"ZIP64 end record is on disk {disk_number}, "
            f"but only {len(reader.parts)} parts were found."
        )
    if disk_count > len(reader.parts):
        raise zipfile.BadZipFile(
            f"Archive expects {disk_count} split parts, "
            f"but only {len(reader.parts)} were found."
        )

    zip64_offset = reader.part_offsets[disk_number] + relative_offset
    reader.seek(zip64_offset)
    data = reader.read(zipfile.sizeEndCentDir64)
    if len(data) != zipfile.sizeEndCentDir64:
        raise OSError("Unable to read ZIP64 end-of-central-directory record")
    if not data.startswith(zipfile.stringEndArchive64):
        raise zipfile.BadZipFile("ZIP64 end-of-central-directory record not found")

    (
        signature,
        _record_size,
        _create_version,
        _read_version,
        current_disk,
        central_directory_disk,
        entries_on_disk,
        entries_total,
        central_directory_size,
        central_directory_offset,
    ) = struct.unpack(zipfile.structEndArchive64, data)

    end_record[zipfile._ECD_SIGNATURE] = signature
    end_record[zipfile._ECD_DISK_NUMBER] = current_disk
    end_record[zipfile._ECD_DISK_START] = central_directory_disk
    end_record[zipfile._ECD_ENTRIES_THIS_DISK] = entries_on_disk
    end_record[zipfile._ECD_ENTRIES_TOTAL] = entries_total
    end_record[zipfile._ECD_SIZE] = central_directory_size
    end_record[zipfile._ECD_OFFSET] = central_directory_offset
    end_record[zipfile._ECD_LOCATION] = zip64_offset
    return end_record


def normalize_end_record_location(reader: SplitZipFile, end_record: list) -> None:
    central_directory_disk = end_record[zipfile._ECD_DISK_START]
    if central_directory_disk >= len(reader.parts):
        raise zipfile.BadZipFile(
            f"Central directory starts on disk {central_directory_disk}, "
            f"but only {len(reader.parts)} parts were found."
        )
    central_directory_base = reader.part_offsets[central_directory_disk]
    end_record[zipfile._ECD_LOCATION] = (
        central_directory_base
        + end_record[zipfile._ECD_OFFSET]
        + end_record[zipfile._ECD_SIZE]
    )


def patch_split_offsets(
    zip_handle: zipfile.ZipFile,
    reader: SplitZipFile,
    end_record,
) -> None:
    if len(reader.parts) == 1:
        return

    concat = (
        end_record[zipfile._ECD_LOCATION]
        - end_record[zipfile._ECD_SIZE]
        - end_record[zipfile._ECD_OFFSET]
    )

    for info in zip_handle.infolist():
        relative_offset = info.header_offset - concat
        if info.volume >= len(reader.part_offsets) - 1:
            raise zipfile.BadZipFile(
                f"{info.filename} starts on split volume {info.volume}, "
                f"but only {len(reader.parts)} archive parts were found."
            )
        info.header_offset = reader.part_offsets[info.volume] + relative_offset

    end_offset = zip_handle.start_dir
    for info in sorted(
        zip_handle.filelist,
        key=lambda zip_info: zip_info.header_offset,
        reverse=True,
    ):
        info._end_offset = end_offset
        end_offset = info.header_offset


def safe_output_path(output_root: Path, member_name: str) -> Path:
    root = output_root.resolve()
    target = (output_root / member_name).resolve()
    if os.path.commonpath([root, target]) != str(root):
        raise ValueError(f"Archive member escapes output directory: {member_name}")
    return target


def extract_member(
    zip_handle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    output_root: Path,
    overwrite: bool,
    chunk_size: int,
) -> str:
    target = safe_output_path(output_root, info.filename)
    if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return "dir"

    if target.exists() and not overwrite and target.stat().st_size == info.file_size:
        return "skip"

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(target.name + ".tmp-extract")
    if temp_target.exists():
        temp_target.unlink()

    with zip_handle.open(info) as source, temp_target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=chunk_size)

    os.replace(temp_target, target)
    try:
        mtime = time.mktime(info.date_time + (0, 0, -1))
        os.utime(target, (mtime, mtime))
    except (OverflowError, ValueError, OSError):
        pass
    return "file"


def extract_archive(
    archive_path: Path,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    chunk_size: int,
) -> None:
    print(f"\nArchive: {archive_path}")
    print(f"Output:  {output_root}")
    zip_handle, reader = open_zip(archive_path)
    try:
        members = zip_handle.infolist()
        file_count = sum(not info.is_dir() for info in members)
        total_size = sum(info.file_size for info in members if not info.is_dir())
        print(f"Members: {len(members)} total, {file_count} files, {total_size:,} bytes")
        if dry_run:
            return

        extracted = skipped = directories = 0
        for index, info in enumerate(members, start=1):
            result = extract_member(zip_handle, info, output_root, overwrite, chunk_size)
            if result == "file":
                extracted += 1
            elif result == "skip":
                skipped += 1
            else:
                directories += 1

            if index == len(members) or index % 1000 == 0:
                print(
                    f"  {index}/{len(members)} members processed "
                    f"({extracted} files extracted, {skipped} skipped, "
                    f"{directories} dirs)"
                )
    finally:
        zip_handle.close()
        reader.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PKU-AIGI-500K ZIP and split-ZIP archives without sudo."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        type=Path,
        help="Directory containing the downloaded PKU-AIGI-500K archives.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=list(ARCHIVE_GROUPS),
        choices=list(ARCHIVE_GROUPS),
        help="Archive groups to extract. Defaults to all groups.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open archives and print counts without extracting files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files even when their uncompressed size matches.",
    )
    parser.add_argument(
        "--chunk-size",
        default=8 * 1024 * 1024,
        type=int,
        help="Copy chunk size in bytes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    for group_name in args.only:
        for archive_name, output_name in ARCHIVE_GROUPS[group_name]["archives"]:
            extract_archive(
                archive_path=data_root / archive_name,
                output_root=data_root / output_name,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
