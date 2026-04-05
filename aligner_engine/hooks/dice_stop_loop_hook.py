import os
import json
import time
from pathlib import Path
from typing import Optional, Union, Sequence, Dict

from mmengine import HOOKS, join_path, HistoryBuffer
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner

import aligner_engine.utils as util

@HOOKS.register_module()
class DICEStopLoopHook(Hook):
    msg = "Training quit cause of button click interrupt"

    def __init__(self):
        self._quit = False

    def quit(self):
        self._quit = True

    def _check_quit_or_not(self):
        if self._quit:
            raise Exception(self.msg)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        self._check_quit_or_not()

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        self._check_quit_or_not()

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        self._check_quit_or_not()

    def after_train_epoch(self, runner) -> None:
        self._check_quit_or_not()

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        self._check_quit_or_not()

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        self._check_quit_or_not()


