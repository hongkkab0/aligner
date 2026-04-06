"""DiceCopyPaste — Copy-Paste augmentation for RotatedBoxes.

Label-integrity guarantee
--------------------------
For every pasted instance, ONLY the center (cx, cy) changes by the paste
offset.  Width, height, and angle are copied verbatim from the source:

    new_cx    = src_cx + (paste_x - src_crop_x1)
    new_cy    = src_cy + (paste_y - src_crop_y1)
    new_w     = src_w       # unchanged
    new_h     = src_h       # unchanged
    new_angle = src_angle   # unchanged

This is the only geometrically correct transformation: a translation does
not change the shape or orientation of a rotated bounding box.

Thread safety
-------------
Each DataLoader worker process owns its own transform instance and
therefore its own cache.  No shared state is involved.
"""

from __future__ import annotations

import random as _random
from typing import Optional

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
from mmrotate.structures import RotatedBoxes


@TRANSFORMS.register_module()
class DiceCopyPaste(BaseTransform):
    """Copy-Paste for RotatedBoxes detection.

    Maintains an internal rolling cache of recently processed
    (image, gt_bboxes, gt_bboxes_labels, gt_ignore_flags) tuples.
    On each call (with probability ``prob``), it randomly selects up to
    ``max_num_pasted`` instances from a cached image and pastes their
    axis-aligned crop region onto the current image, then appends the
    corresponding RotatedBox annotations with the correct translated center.

    Args:
        prob (float): Probability of applying the augmentation. Default: 0.3
        max_num_pasted (int): Maximum instances pasted per image. Default: 10
        cache_size (int): Rolling cache capacity. Default: 20
        min_bbox_side (int): Skip instances whose axis-aligned crop is smaller
            than this value on either side (avoids pasting invisible dots).
            Default: 4
    """

    def __init__(
        self,
        prob: float = 0.3,
        max_num_pasted: int = 10,
        cache_size: int = 20,
        min_bbox_side: int = 4,
    ) -> None:
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.max_num_pasted = max_num_pasted
        self.cache_size = cache_size
        self.min_bbox_side = min_bbox_side
        self._cache: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_cache(self, results: dict) -> None:
        """Snapshot the current result into the rolling cache."""
        bboxes = results.get("gt_bboxes")
        if bboxes is None or len(bboxes) == 0:
            return
        ignore = results.get("gt_ignore_flags")
        if ignore is None:
            ignore = np.zeros(len(bboxes), dtype=bool)
        self._cache.append({
            "img":              results["img"].copy(),
            "gt_bboxes":        bboxes.clone(),
            "gt_bboxes_labels": results["gt_bboxes_labels"].copy(),
            "gt_ignore_flags":  ignore.copy(),
        })
        if len(self._cache) > self.cache_size:
            self._cache.pop(0)

    @staticmethod
    def _axis_aligned_crop(rbox_tensor: torch.Tensor,
                           img_h: int, img_w: int
                           ) -> Optional[tuple[int, int, int, int]]:
        """Return integer (x1, y1, x2, y2) of the axis-aligned bounding rect
        of a RotatedBox, clamped to the image.  Returns None if degenerate."""
        corners = RotatedBoxes.rbox2corner(rbox_tensor)   # (4, 2)
        x1 = int(max(0.0,   corners[:, 0].min().item()))
        y1 = int(max(0.0,   corners[:, 1].min().item()))
        x2 = int(min(float(img_w), corners[:, 0].max().item()))
        y2 = int(min(float(img_h), corners[:, 1].max().item()))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    # ------------------------------------------------------------------
    # Main transform
    # ------------------------------------------------------------------

    def transform(self, results: dict) -> dict:
        # Always cache the incoming sample first (before any modification so
        # future pastes use clean, fully-augmented images from earlier steps).
        self._update_cache(results)

        if _random.random() >= self.prob or not self._cache:
            return results

        src = _random.choice(self._cache)
        src_img    = src["img"]
        src_bboxes = src["gt_bboxes"]        # RotatedBoxes
        src_labels = src["gt_bboxes_labels"]
        src_ignore = src["gt_ignore_flags"]

        dst_img = results["img"]
        dst_h, dst_w = dst_img.shape[:2]
        src_h, src_w = src_img.shape[:2]

        n_src = len(src_bboxes)
        if n_src == 0:
            return results

        n_paste = min(n_src, self.max_num_pasted)
        indices = np.random.choice(n_src, n_paste, replace=False)

        new_rboxes:  list[torch.Tensor] = []
        new_labels:  list               = []
        new_ignores: list               = []

        for idx in indices:
            rbox = src_bboxes.tensor[idx]   # [cx, cy, w, h, angle]

            crop = self._axis_aligned_crop(rbox, src_h, src_w)
            if crop is None:
                continue
            sx1, sy1, sx2, sy2 = crop

            ph = sy2 - sy1
            pw = sx2 - sx1
            if ph < self.min_bbox_side or pw < self.min_bbox_side:
                continue

            # Random paste position that keeps the whole patch inside dst
            max_px = dst_w - pw
            max_py = dst_h - ph
            if max_px <= 0 or max_py <= 0:
                continue

            px = np.random.randint(0, max_px)
            py = np.random.randint(0, max_py)

            # ── Label-integrity: ONLY cx/cy change, by the translation vector ──
            offset_x = float(px - sx1)
            offset_y = float(py - sy1)
            new_cx = rbox[0].item() + offset_x
            new_cy = rbox[1].item() + offset_y

            # Skip if translated center falls outside destination (can happen
            # when source bbox center was near the edge of the source crop).
            if not (0.0 <= new_cx < dst_w and 0.0 <= new_cy < dst_h):
                continue

            new_rbox = torch.tensor(
                [new_cx, new_cy, rbox[2].item(), rbox[3].item(), rbox[4].item()],
                dtype=rbox.dtype,
            )

            # Paste pixel data (rectangular crop — background included, which
            # is acceptable and standard for Copy-Paste augmentation).
            dst_img[py: py + ph, px: px + pw] = src_img[sy1: sy2, sx1: sx2]

            new_rboxes.append(new_rbox)
            new_labels.append(src_labels[idx])
            new_ignores.append(src_ignore[idx])

        if new_rboxes:
            stacked = torch.stack(new_rboxes, dim=0)
            results["gt_bboxes"] = RotatedBoxes(
                torch.cat([results["gt_bboxes"].tensor, stacked], dim=0)
            )
            label_dtype = results["gt_bboxes_labels"].dtype
            results["gt_bboxes_labels"] = np.concatenate([
                results["gt_bboxes_labels"],
                np.array(new_labels, dtype=label_dtype),
            ])
            if results.get("gt_ignore_flags") is not None:
                results["gt_ignore_flags"] = np.concatenate([
                    results["gt_ignore_flags"],
                    np.array(new_ignores, dtype=bool),
                ])

        results["img"] = dst_img
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prob={self.prob}, "
            f"max_num_pasted={self.max_num_pasted}, "
            f"cache_size={self.cache_size}, "
            f"min_bbox_side={self.min_bbox_side})"
        )
