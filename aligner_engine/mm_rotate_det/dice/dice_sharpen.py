from typing import Dict, Tuple

import cv2
from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
import numpy as np
import random


@TRANSFORMS.register_module()
class DiceSharpen(BaseTransform):
    """Apply gaussian noise to the input image.

    Args:
        alpha: Default: (0.2, 0.5).
        lightness: Default: (0.5, 1.0).
        prob: probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
            self,
            alpha: Tuple[float, float] = (0.2, 0.5),
            lightness: Tuple[float, float] = (0.5, 1.0),
            prob: float = 0.5
    ) -> None:
        if isinstance(alpha, tuple):
            if alpha[0] < 0:
                msg = "Lower alpha should be non negative."
                raise ValueError(msg)
            if alpha[1] < 0:
                msg = "Upper alpha should be non negative."
                raise ValueError(msg)
        else:
            msg = "alpha should be tuple."
            raise ValueError(msg)
        self.alpha = alpha
        self.lightness = lightness
        self.p = prob

    def _generate_sharpening_matrix(self) -> np.ndarray:
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)

        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]],
            dtype=np.float32,
        )
        return (1 - alpha) * matrix_nochange + alpha * matrix_effect

    def transform(self, results: dict) -> dict:
        image = results['img']
        sharpen_filter = self._generate_sharpening_matrix()
        sharpen_img = cv2.filter2D(image, -1, sharpen_filter)
        results['img'] = sharpen_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alpha={self.alpha}, '
        repr_str += f'lightness={self.lightness}, '
        repr_str += f'p={self.p}'
        return repr_str
