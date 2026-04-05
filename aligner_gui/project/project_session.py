from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, List


class ProjectSession:
    def __init__(self, project_path: str, is_new: bool):
        from aligner_engine.default_project_config import get_default_project_config
        from aligner_engine.model_profiles import get_model_profile
        from aligner_engine.project_settings import ProjectSettings
        import aligner_engine.const as const
        import aligner_engine.utils as util

        self.project_path = project_path
        self.is_new = is_new
        self._worker = None
        self._lock = RLock()
        self._default_project_config_factory = get_default_project_config
        self._default_model_profile_id = get_model_profile(None).id
        self._settings_cls = ProjectSettings
        self._engine_const = const
        self._engine_util = util
        self._project_settings, self._project_config = self._load_settings_from_disk()

    def _create_worker(self):
        from aligner_engine.worker import Worker

        return Worker(self.project_path, is_new=self.is_new)

    def _load_settings_from_disk(self):
        project_config_path = self._engine_util.join_path(self.project_path, self._engine_const.PROJECT_CONFIG_NAME)
        merged = self._default_project_config_factory()
        try:
            if self._engine_util.is_exist(project_config_path):
                project_config = self._engine_util.read_yaml(project_config_path)
                if project_config is not None:
                    merged.update(project_config)
        except Exception:
            pass

        settings = self._settings_cls.from_dict(
            merged,
            default_model_profile=self._default_model_profile_id,
        )
        return deepcopy(settings), settings.to_dict()

    @property
    def worker(self):
        with self._lock:
            if self._worker is None:
                self._worker = self._create_worker()
            return self._worker

    def get_model_profiles(self) -> List[Any]:
        from aligner_engine.model_profiles import list_model_profiles

        return list_model_profiles()

    def get_project_config(self):
        with self._lock:
            return deepcopy(self._project_config)

    def get_project_settings(self):
        with self._lock:
            return deepcopy(self._project_settings)

    def set_project_config(self, project_config):
        settings = self._settings_cls.from_dict(
            project_config,
            default_model_profile=self._default_model_profile_id,
        )
        self.set_project_settings(settings)

    def set_project_settings(self, settings):
        with self._lock:
            worker = self._worker
            self._project_settings = deepcopy(settings)
            self._project_config = settings.to_dict()

        if worker is not None:
            worker.set_project_settings(settings)
            return

        project_config_path = self._engine_util.join_path(self.project_path, self._engine_const.PROJECT_CONFIG_NAME)
        self._engine_util.write_yaml(project_config_path, settings.to_dict())

    def get_dataset_summary_path(self):
        return self._engine_util.join_path(self.project_path, "labeler_dataset_summary.json")

    def close(self):
        with self._lock:
            worker = self._worker
            self._worker = None

        if worker is None:
            return

        for method_name in ("stop_test", "stop_training", "close_logger"):
            method = getattr(worker, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception:
                continue

    def __getattr__(self, item: str):
        return getattr(self.worker, item)
