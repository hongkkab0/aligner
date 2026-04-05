from typing import Union, Tuple, List

from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
from mmcv.transforms.utils import cache_randomness
from shapely.geometry import Polygon
import numpy as np
from mmrotate.structures import RotatedBoxes
from shapely.validation import make_valid
import random


def make_valid(polygon):
    from shapely.geometry.base import geom_factory
    from shapely.geos import lgeos

    if polygon.is_valid:
        return polygon

    return geom_factory(lgeos.GEOSMakeValid(polygon._geom))


@TRANSFORMS.register_module()
class DiceRandomErasing(BaseTransform):
    """RandomErasing operation.
    Random Erasing randomly selects a rectangle region
    in an image and erases its pixels with random values.
    `RandomErasing <https://arxiv.org/abs/1708.04896>`_.
    Required Keys:
    - img
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)
    Modified Keys:
    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)
    Args:
        n_patches (int or tuple[int, int]): Number of regions to be dropped.
            If it is given as a tuple, number of patches will be randomly
            selected from the closed interval [``n_patches[0]``,
            ``n_patches[1]``].
        ratio (float or tuple[float, float]): The ratio of erased regions.
            It can be ``float`` to use a fixed ratio or ``tuple[float, float]``
            to randomly choose ratio from the interval.
        squared (bool): Whether to erase square region. Defaults to True.
        bbox_erased_thr (float): The threshold for the maximum area proportion
            of the bbox to be erased. When the proportion of the area where the
            bbox is erased is greater than the threshold, the bbox will be
            removed. Defaults to 0.9.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
    """

    def __init__(
            self,
            n_patches: Union[int, Tuple[int, int]],
            ratio: Union[float, Tuple[float, float]],
            squared: bool = True,
            img_border_value: Union[int, float, tuple, str] = 'random',
            mask_border_value: int = 0,
            seg_ignore_label: int = 255,
    ) -> None:
        if isinstance(n_patches, tuple):
            assert len(n_patches) == 2 and 0 <= n_patches[0] < n_patches[1]
        else:
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, tuple):
            assert len(ratio) == 2 and 0 <= ratio[0] < ratio[1] <= 1
        else:
            ratio = (ratio, ratio)

        self.n_patches = n_patches
        self.ratio = ratio
        self.squared = squared
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label

    @cache_randomness
    def _get_patches(self, img_shape: Tuple[int, int]) -> List[list]:
        """Get patches for random erasing."""
        patches = []
        n_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)
        for _ in range(n_patches):
            if self.squared:
                ratio = np.random.random() * (self.ratio[1] -
                                              self.ratio[0]) + self.ratio[0]
                ratio = (ratio, ratio)
            else:
                ratio = (np.random.random() * (self.ratio[1] - self.ratio[0]) +
                         self.ratio[0], np.random.random() *
                         (self.ratio[1] - self.ratio[0]) + self.ratio[0])
            ph, pw = int(img_shape[0] * ratio[0]), int(img_shape[1] * ratio[1])
            px1, py1 = np.random.randint(0,
                                         img_shape[1] - pw), np.random.randint(
                0, img_shape[0] - ph)
            px2, py2 = px1 + pw, py1 + ph
            patches.append([px1, py1, px2, py2])
        return np.array(patches)

    def _transform_img(self, results: dict, patches: List[list], patches_valid: List[bool]) -> None:
        """Random erasing the image."""
        for patch in patches[patches_valid]:
            px1, py1, px2, py2 = patch
            var = random.uniform(10, 50)
            sigma = var ** 0.5
            mean = 128
            if self.img_border_value == 'random':
                eraser = np.random.normal(mean, sigma, results['img'][py1:py2, px1:px2, :].shape)
                results['img'][py1:py2, px1:px2, :] = eraser
                results['img'] = np.clip(results['img'], 0, 255)
                results['img'] = results['img'].astype(np.uint8)
            else:
                results['img'][py1:py2, px1:px2, :] = self.img_border_value

    def _transform_bboxes(self, results: dict, patches: List[list]) -> List[bool]:
        """Random erasing the bboxes."""
        bboxes = results['gt_bboxes']
        patches_valid = [True for i in range(len(patches))]

        patches_polygon = []
        for [px1, py1, px2, py2] in patches:
            rect = Polygon([(px1, py1), (px1, py2), (px2, py2), (px2, py1)])
            if not rect.is_valid:
                make_valid(rect)
            patches_polygon.append(rect)

        bboxes_poly = []
        for rbox in bboxes.tensor:
            bbox = Polygon((RotatedBoxes.rbox2corner(rbox.clone().detach())).tolist())
            if not bbox.is_valid:
                make_valid(bbox)
            bboxes_poly.append(bbox)

        for bbox in bboxes_poly:
            for i, patch in enumerate(patches_polygon):
                if bbox.intersects(patch):
                    patches_valid[i] = False

        return patches_valid

    def transform(self, results: dict) -> dict:
        """Transform function to erase some regions of image."""
        patches = self._get_patches(results['img_shape'])
        if results.get('gt_bboxes', None) is not None:
            patches_valid = self._transform_bboxes(results, patches)
        self._transform_img(results, patches, patches_valid)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_patches={self.n_patches}, '
        repr_str += f'ratio={self.ratio}, '
        repr_str += f'squared={self.squared}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str
