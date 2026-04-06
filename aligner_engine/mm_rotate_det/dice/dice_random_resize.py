"""DiceRandomResize — Scale-jitter resize that is safe with RotatedBoxes.

Why this exists
---------------
``mmdet.RandomResize`` inherits from ``mmcv.transforms.RandomResize``.
mmcv's internal ``Resize._resize_bboxes`` uses the legacy
``gt_bboxes * np.tile(...)`` approach, which raises::

    TypeError: unsupported operand type(s) for *: 'RotatedBoxes' and 'float'

``mmdet.Resize`` *does* override ``_resize_bboxes`` to use the mmdetection
box-API (``bboxes.rescale_()``), which is polymorphic and handles
RotatedBoxes correctly.

This transform picks a random target size from ``[scale[0], scale[1]]``
and delegates the actual resize to ``mmdet.Resize``, bypassing the broken
mmcv path entirely.
"""
from __future__ import annotations

import random as _random

from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DiceRandomResize(BaseTransform):
    """Scale-jitter resize compatible with RotatedBoxes.

    Args:
        scale (list[tuple[int,int]]): Two ``(height, width)`` tuples that
            define the inclusive range for random scale selection, e.g.
            ``[(256, 256), (1024, 1024)]``.
        keep_ratio (bool): Whether to keep the aspect ratio. Default: True.
    """

    def __init__(
        self,
        scale: list,
        keep_ratio: bool = True,
    ) -> None:
        assert len(scale) == 2, "scale must be [(min_h, min_w), (max_h, max_w)]"
        self.scale = scale
        self.keep_ratio = keep_ratio

    def transform(self, results: dict) -> dict:
        min_s, max_s = self.scale
        h = _random.randint(min_s[0], max_s[0])
        w = _random.randint(min_s[1], max_s[1])

        # mmdet.Resize overrides _resize_bboxes to use bboxes.rescale_()
        # which works correctly with RotatedBoxes.
        from mmdet.datasets.transforms import Resize as MmdetResize
        return MmdetResize(scale=(h, w), keep_ratio=self.keep_ratio)(results)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale={self.scale}, keep_ratio={self.keep_ratio})"
        )
