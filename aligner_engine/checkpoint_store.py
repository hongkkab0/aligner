from __future__ import annotations

from dataclasses import dataclass

import aligner_engine.const as const
import aligner_engine.utils as util


@dataclass
class CheckpointSaveSummary:
    succeeded: list[str]
    failed: list[str]


class ProjectCheckpointStore:
    def __init__(self, project_path: str):
        self._project_path = project_path

    def save(self, ckpt_path: str, save_dir: str) -> bool:
        dst_path = util.join_path(self._project_path, const.DIRNAME_AUTOSAVED, save_dir)
        util.make_dir(dst_path)
        result = util.copy_file(ckpt_path, dst_path)
        result = result and util.copy_file(
            util.join_path(self._project_path, const.FILENAME_MODEL_CONFIG),
            util.join_path(self._project_path, const.DIRNAME_AUTOSAVED, const.FILENAME_MODEL_CONFIG),
        )
        return bool(result)

    @staticmethod
    def summarize(results: list[tuple[str, bool]]) -> CheckpointSaveSummary:
        succeeded = [name for name, ok in results if ok]
        failed = [name for name, ok in results if not ok]
        return CheckpointSaveSummary(succeeded=succeeded, failed=failed)
