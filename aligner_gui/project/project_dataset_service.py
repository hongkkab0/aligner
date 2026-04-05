from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

from aligner_gui.labeler.libs.labelFile import LabelFile


LABELER_IMAGE_LIST_FILE_NAME = "labeler_image_list.json"


@dataclass
class ImageLabelState:
    path: str
    has_label: bool
    is_empty: bool
    needs_confirm: bool
    labels: List[str]


def get_labeler_image_list_path(project_path: str) -> str:
    return os.path.join(project_path, LABELER_IMAGE_LIST_FILE_NAME)


def save_labeler_image_list(project_path: str, image_paths: Iterable[str]) -> None:
    with open(get_labeler_image_list_path(project_path), "w", encoding="utf-8") as f:
        json.dump({"image_paths": list(image_paths)}, f, ensure_ascii=False, indent=2)


def load_labeler_image_list(project_path: str) -> List[str]:
    try:
        with open(get_labeler_image_list_path(project_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return list(data.get("image_paths", []))


def inspect_image_labels(
    image_paths: Iterable[str],
    progress_callback=None,
    should_stop=None,
) -> List[ImageLabelState]:
    states: List[ImageLabelState] = []
    image_paths = list(image_paths)
    total = len(image_paths)
    for idx, image_path in enumerate(image_paths):
        if should_stop is not None and should_stop():
            break
        if not os.path.isfile(image_path):
            if progress_callback is not None:
                progress_callback(idx + 1, total, image_path)
            continue

        label_file = LabelFile(image_path)
        if not label_file.existLabel():
            states.append(
                ImageLabelState(
                    path=image_path,
                    has_label=False,
                    is_empty=False,
                    needs_confirm=False,
                    labels=[],
                )
            )
            if progress_callback is not None:
                progress_callback(idx + 1, total, image_path)
            continue

        label_file.load_label()
        shapes = label_file.get_shapes()
        states.append(
            ImageLabelState(
                path=image_path,
                has_label=True,
                is_empty=len(shapes) == 0,
                needs_confirm=label_file.get_is_need_confirm(),
                labels=[shape.get_label() for shape in shapes],
            )
        )
        if progress_callback is not None:
            progress_callback(idx + 1, total, image_path)
    return states


def build_dataset_summary(image_paths: Iterable[str], dataset_summary_path: str, include_empty: bool) -> bool:
    try:
        data_summary = []
        not_empty_data_summary = []
        classes = {}

        for image_state in inspect_image_labels(image_paths):
            if not image_state.has_label:
                continue

            label_path = LabelFile(image_state.path).getLabelPath()
            for class_name in image_state.labels:
                classes[class_name] = class_name

            if image_state.is_empty:
                if include_empty:
                    data_summary.append({"img_path": image_state.path, "label_path": label_path})
            else:
                item = {"img_path": image_state.path, "label_path": label_path}
                data_summary.append(item)
                not_empty_data_summary.append(item)

        if len(not_empty_data_summary) < 10:
            logging.error("Dataset is too small. You have to prepare non-empty labeled images more than 10.")
            raise ValueError("Too small dataset")

        class_summary = {
            "num_classes": len(classes),
            "classes": [
                {"name": class_name, "idx": idx, "color": "#000000"}
                for idx, class_name in enumerate(classes.keys())
            ],
        }

        dataset_summary = {
            "task_type": LabelFile.TASK_TYPE_ROTATE_DET,
            "class_summary": class_summary,
            "data_summary": data_summary,
        }
        with open(dataset_summary_path, "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=4)

    except Exception as exc:
        logging.error("ERROR - %s", exc)
        raise

    return True


def build_dataset_summary_from_project(project_path: str, dataset_summary_path: str, include_empty: bool) -> bool:
    image_paths = load_labeler_image_list(project_path)
    if len(image_paths) == 0:
        raise FileNotFoundError("No indexed images were found for this project.")
    return build_dataset_summary(image_paths, dataset_summary_path, include_empty)
