from __future__ import annotations
import typing as tp
from pathlib import Path

CONST_SUPPORTED_IMAGE_SUFFIXES: tp.Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
)


def iter_chunks(lst: tp.List[tp.Any], int_n: int) -> tp.Iterator[tp.List[tp.Any]]:
    """Yield successive n-sized chunks from lst."""
    for int_i in range(0, len(lst), int_n):
        yield lst[int_i: int_i + int_n]


def collect_input_groups(
    path_input: Path,
) -> tp.List[tp.Tuple[str, tp.List[Path]]]:
    """Collect image groups from a file or directory.

    Single file -> one group (group_id = stem).
    Directory with subdirs -> one group per subdir (IMG_ID pattern).
    Flat directory -> one group named 'batch'.

    Returns list of (str_groupId, list_imagePaths).
    """
    if path_input.is_file():
        if path_input.suffix.lower() not in CONST_SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"지원하지 않는 이미지 형식: {path_input.suffix}")
        if not path_input.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path_input}")
        return [(path_input.stem, [path_input])]

    if not path_input.exists():
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {path_input}")

    list_subdirs = sorted([p for p in path_input.iterdir() if p.is_dir()])
    if list_subdirs:
        list_groups: tp.List[tp.Tuple[str, tp.List[Path]]] = []
        for path_sub in list_subdirs:
            list_imgs = sorted([
                p for p in path_sub.iterdir()
                if p.is_file() and p.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
            ])
            if list_imgs:
                list_groups.append((path_sub.name, list_imgs))
        if list_groups:
            return list_groups

    list_imgs = sorted([
        p for p in path_input.iterdir()
        if p.is_file() and p.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
    ])
    if not list_imgs:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path_input}")
    return [("batch", list_imgs)]


def build_image_output_dir(
    path_outputRoot: Path,
    str_groupId: str,
    path_image: Path,
    bool_isBatch: bool,
) -> Path:
    """Return per-image output directory path."""
    if bool_isBatch:
        return path_outputRoot / str_groupId / path_image.stem
    return path_outputRoot
