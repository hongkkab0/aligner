from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from shutil import copyfile
from typing import List


EXPORT_METADATA_NAME = "export_metadata.json"


@dataclass(frozen=True)
class ExportCheckpointOption:
    source_name: str
    display_name: str
    path: str


def _load_engine_modules():
    import aligner_engine.const as engine_const
    import aligner_engine.utils as engine_util
    return engine_const, engine_util


def _load_export_backend():
    try:
        from aligner_engine.deploy_oepnvino import deploy_openvino
    except ImportError:
        from aligner_engine.deploy_openvio import deploy_openvino
    return deploy_openvino


def list_export_checkpoint_options(project_path: str) -> List[ExportCheckpointOption]:
    engine_const, engine_util = _load_engine_modules()

    options = []
    best_path = engine_util.join_path(project_path, engine_const.DIRNAME_AUTOSAVED, engine_const.FILENAME_CKPT)
    last_path = engine_util.join_path(project_path, engine_const.DIRNAME_AUTOSAVED, engine_const.FILENAME_CKPT_LAST)

    if os.path.isfile(best_path):
        options.append(
            ExportCheckpointOption(
                source_name=engine_const.FILENAME_CKPT,
                display_name="Best checkpoint",
                path=best_path,
            )
        )
    if os.path.isfile(last_path):
        options.append(
            ExportCheckpointOption(
                source_name=engine_const.FILENAME_CKPT_LAST,
                display_name="Last checkpoint",
                path=last_path,
            )
        )
    return options


def export_project_bundle(
    project_path: str,
    export_path: str,
    checkpoint_option: ExportCheckpointOption,
    processing_signal,
    project_config: dict | None = None,
) -> bool:
    engine_const, engine_util = _load_engine_modules()
    deploy_openvino = _load_export_backend()

    temp_paths = []
    try:
        logging.info("Export started. checkpoint=%s", checkpoint_option.source_name)
        processing_signal.emit(1, "Preparing files...")

        cfg_model_path_src = engine_util.join_path(
            project_path,
            engine_const.DIRNAME_AUTOSAVED,
            engine_const.FILENAME_MODEL_CONFIG,
        )
        cfg_model_path_temp = engine_util.join_path(project_path, "export_" + engine_const.FILENAME_MODEL_CONFIG)
        copyfile(cfg_model_path_src, cfg_model_path_temp)
        temp_paths.append(cfg_model_path_temp)

        ckpt_path_temp = engine_util.join_path(project_path, "export_" + engine_const.FILENAME_CKPT)
        copyfile(checkpoint_option.path, ckpt_path_temp)
        temp_paths.append(ckpt_path_temp)

        sample_img_path = os.path.join(project_path, engine_const.FILENAME_SAMPLE_IMAGE_FOR_DEPLOYMENT)
        work_dir = os.path.join(project_path, engine_const.DIRNAME_DEPLOYMENT_WORK_DIR)

        deploy_openvino(cfg_model_path_temp, ckpt_path_temp, sample_img_path, work_dir, processing_signal)

        processing_signal.emit(5, "Copying files...")
        copyfile(
            cfg_model_path_temp,
            engine_util.join_path(export_path, engine_const.FILENAME_MODEL_CONFIG),
        )
        copyfile(
            ckpt_path_temp,
            engine_util.join_path(export_path, engine_const.FILENAME_CKPT),
        )
        copyfile(
            engine_util.join_path(work_dir, engine_const.FILENAME_DEPLOY_CONFIG),
            engine_util.join_path(export_path, engine_const.FILENAME_DEPLOY_CONFIG),
        )
        copyfile(
            engine_util.join_path(work_dir, engine_const.FILENAME_EXPORT_VINO_BIN),
            engine_util.join_path(export_path, engine_const.FILENAME_EXPORT_VINO_BIN),
        )
        copyfile(
            engine_util.join_path(work_dir, engine_const.FILENAME_EXPORT_VINO_XML),
            engine_util.join_path(export_path, engine_const.FILENAME_EXPORT_VINO_XML),
        )

        metadata = {
            "selected_checkpoint": checkpoint_option.source_name,
            "exported_checkpoint_name": engine_const.FILENAME_CKPT,
            "model_profile": "" if project_config is None else project_config.get("model.profile", ""),
            "openvino_enabled": True if project_config is None else bool(project_config.get("inference.enable_openvino", True)),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(engine_util.join_path(export_path, EXPORT_METADATA_NAME), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        processing_signal.emit(6, "Export succeeded.")
        logging.info("Export succeeded.")
        return True
    except Exception as exc:
        logging.exception("Export failed: %s", exc)
        return False
    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    continue
