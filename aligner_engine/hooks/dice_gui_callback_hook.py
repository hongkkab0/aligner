import time
from typing import Optional, Union, Sequence, Dict, List

from mmengine import HOOKS, join_path, HistoryBuffer
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner

import aligner_engine.const as const


@HOOKS.register_module()
class DICECallbackHook(Hook):
    priority = 'VERY_LOW'

    def __init__(self, worker, iter_callback, epoch_callback):
        self._worker = worker
        self._iter_callback = iter_callback  # function(str, int, int)
        self._epoch_callback = epoch_callback  # function(int, str)
        self._classes: List = []
        self._last_train_iter_emit_ts = 0.0
        self._last_val_iter_emit_ts = 0.0
        self._iter_emit_interval_sec = 0.2

    def before_train(self, runner) -> None:
        classes = [c['name'] for c in runner.train_dataloader.dataset.class_meta]
        self._classes = classes
        class_index = {v: idx for idx, v in enumerate(classes)}
        class_name = {idx: v for idx, v in enumerate(classes)}
        self._worker.set_train_val_summary_class(class_index, class_name)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        max_iter = len(runner.train_loop.dataloader)
        cur_iter = runner.iter % max_iter if max_iter > 0 else -1
        now = time.monotonic()
        is_last_iter = max_iter > 0 and (cur_iter + 1) >= max_iter
        if cur_iter <= 0 or is_last_iter or (now - self._last_train_iter_emit_ts) >= self._iter_emit_interval_sec:
            self._last_train_iter_emit_ts = now
            self._iter_callback(const.PHASE_TYPE_TRAINING, cur_iter, max_iter)

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        max_iter = len(runner.val_loop.dataloader)
        now = time.monotonic()
        is_last_iter = max_iter > 0 and (batch_idx + 1) >= max_iter
        if batch_idx <= 0 or is_last_iter or (now - self._last_val_iter_emit_ts) >= self._iter_emit_interval_sec:
            self._last_val_iter_emit_ts = now
            self._iter_callback(const.PHASE_TYPE_VALIDATION, batch_idx, max_iter)

    def after_train_epoch(self, runner) -> None:
        max_iter = len(runner.train_loop.dataloader)
        loss = runner.message_hub.log_scalars['train/loss'].mean(max_iter)
        epoch = runner.epoch + 1
        self._worker.set_summary_training_loss(epoch, loss)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        epoch = runner.epoch
        map = metrics["dota/AP75"]
        ap = {idx: [metrics["dota/AP75_detail"][idx]["ap"], metrics["dota/AP75_detail"][idx]["num_gts"]] for idx, c in
              enumerate(self._classes)}
        mpe = metrics["dota/mPE"]
        mpe_by_class = metrics.get("dota/mPE_by_class")
        self._worker.set_summary_validation_result(epoch, map, ap, mpe, mpe_by_class)
        self._epoch_callback(epoch, runner.message_hub.runtime_info['last_ckpt'])
