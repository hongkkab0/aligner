from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class FileListRemovalResult:
    image_paths: List[str]
    labeled_info: Dict[str, bool]
    labeled_count: int
    removed_count: int
    removed_current: bool
    next_image_path: str | None


def remove_paths_from_file_list(
    image_paths: Iterable[str],
    labeled_info: Dict[str, bool],
    target_paths: Iterable[str],
    current_image_path: str | None,
) -> FileListRemovalResult:
    current_paths = list(image_paths)
    removed_paths = {path for path in target_paths}
    updated_labeled_info = dict(labeled_info)
    labeled_count = sum(1 for _, is_labeled in updated_labeled_info.items() if is_labeled)

    try:
        current_index = current_paths.index(current_image_path) if current_image_path is not None else -1
    except ValueError:
        current_index = -1

    for removed_path in removed_paths:
        if updated_labeled_info.get(removed_path, False):
            labeled_count -= 1
        updated_labeled_info.pop(removed_path, None)

    updated_paths = [path for path in current_paths if path not in removed_paths]
    removed_current = current_image_path in removed_paths
    next_image_path = None
    if removed_current and updated_paths:
        next_index = min(max(current_index, 0), len(updated_paths) - 1)
        next_image_path = updated_paths[next_index]

    return FileListRemovalResult(
        image_paths=updated_paths,
        labeled_info=updated_labeled_info,
        labeled_count=max(labeled_count, 0),
        removed_count=len(removed_paths),
        removed_current=removed_current,
        next_image_path=next_image_path,
    )
