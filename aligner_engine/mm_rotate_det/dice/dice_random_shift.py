from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from numpy import random
from aligner_engine.mm_rotate_det.dice.aug_util import get_bounding_rect_of_rboxes

import numpy as np
@TRANSFORMS.register_module()
class DiceRandomShift(BaseTransform):
    """Shift the image and box given shift pixels and probability.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignore_flags (bool) (optional)

    Args:
        prob (float): Probability of shifts. Defaults to 0.5.
        max_shift_px (int): The max pixels for shifting. Defaults to 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Defaults to 1.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_shift_px: int = 32,
                 filter_thr_px: int = 1,
                 max_shift_percent: float = 0.1) -> None:
        assert 0 <= prob <= 1
        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        self.max_shift_percent = max_shift_percent

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if self._random_prob() < self.prob:
            img_shape = results['img'].shape[:2]
            img_width = img_shape[1]
            img_height = img_shape[0]

            upper_left_of_all_boxes, lower_right_of_all_boxes = get_bounding_rect_of_rboxes(results['gt_bboxes'], img_width, img_height)
            left_of_all_boxes = upper_left_of_all_boxes[0]
            right_of_all_boxes = lower_right_of_all_boxes[0]
            top_of_all_boxes = upper_left_of_all_boxes[1]
            bot_of_all_boxes = lower_right_of_all_boxes[1]

            margin_x = 0
            margin_y = 0
            left_bound = max(0, left_of_all_boxes - margin_x)
            right_bound = max(0, img_width - right_of_all_boxes - margin_x)
            top_bound = max(0, top_of_all_boxes - margin_y)
            bot_bound = max(0, img_height - bot_of_all_boxes - margin_y)

            if right_bound - left_bound >= 1:
                random_shift_x = random.randint(-left_bound, right_bound)
            else:
                random_shift_x = 0

            if bot_bound - top_bound >= 1:
                random_shift_y = random.randint(-top_bound, bot_bound)
            else:
                random_shift_y = 0

            new_x = max(0, random_shift_x)
            ori_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            ori_y = max(0, -random_shift_y)

            # TODO: support mask and semantic segmentation maps.
            bboxes = results['gt_bboxes'].clone()
            bboxes.translate_([random_shift_x, random_shift_y])

            # clip border
            # bboxes.clip_(img_shape)

            # remove invalid bboxes
            valid_inds = (bboxes.widths > self.filter_thr_px).numpy() & (
                bboxes.heights > self.filter_thr_px).numpy()
            # If the shift does not contain any gt-bbox area, skip this
            # image.
            if not valid_inds.any():
                return results
            bboxes = bboxes[valid_inds]
            results['gt_bboxes'] = bboxes
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            # shift img
            img = results['img']
            new_img = np.zeros_like(img)
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(random_shift_y)
            new_w = img_w - np.abs(random_shift_x)
            new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                = img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
            results['img'] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_shift_px={self.max_shift_px}, '
        repr_str += f'filter_thr_px={self.filter_thr_px})'
        return repr_str