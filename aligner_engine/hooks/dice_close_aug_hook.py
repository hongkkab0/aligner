"""DiceCloseAugHook — Augmentation annealing for the final training epochs.

Rationale
---------
Heavy augmentations (Mosaic, Copy-Paste, aggressive scale jitter) improve
generalisation in the early and mid phases of training by exposing the
model to diverse, synthetic-looking samples.

In the final epochs, however, these transforms *hurt* convergence:
- Mosaic assembles four sub-images, so the model rarely sees objects
  at their true inference scale just before evaluation.
- Copy-Paste and scale jitter add noise that prevents the optimiser
  from fine-tuning to the exact feature distributions in real images.

Switching to a lighter pipeline for the last N epochs ("closing" aug)
lets the model settle on clean, inference-like samples and consistently
raises the final mAP — this is the same trick used by YOLOv8/v9.

Behaviour
---------
- At `close_aug_epoch` (0-based), the train-dataloader's dataset pipeline
  is replaced with `close_aug_pipeline` (a plain list-of-dicts).
- The switch happens exactly once (guarded by `_switched`).
- A log message is emitted so the change is visible in the training log.
"""
from __future__ import annotations

from mmengine import HOOKS
from mmengine.hooks import Hook
from mmcv.transforms import Compose


@HOOKS.register_module()
class DiceCloseAugHook(Hook):
    """Replace the training pipeline with a lighter one after a given epoch.

    Args:
        close_aug_epoch (int): 0-based epoch index at which to switch.
            Example: ``close_aug_epoch=60`` switches *before* epoch 61
            starts (with 1-based counting).
        close_aug_pipeline (list[dict]): Full pipeline config to use from
            ``close_aug_epoch`` onwards.  Should omit Mosaic, Copy-Paste,
            and scale-jitter transforms.
    """

    def __init__(
        self,
        close_aug_epoch: int,
        close_aug_pipeline: list,
    ) -> None:
        self.close_aug_epoch = close_aug_epoch
        self.close_aug_pipeline = close_aug_pipeline
        self._switched = False

    def before_train_epoch(self, runner) -> None:
        if self._switched or runner.epoch < self.close_aug_epoch:
            return

        dataset = runner.train_dataloader.dataset
        dataset.pipeline = Compose(self.close_aug_pipeline)
        self._switched = True

        runner.logger.info(
            f"[DiceCloseAugHook] epoch {runner.epoch + 1}: "
            f"switched to light augmentation pipeline "
            f"(Mosaic / Copy-Paste / scale-jitter disabled)"
        )
