from typing import Union, Tuple

from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS
from mmcv.transforms.utils import cache_randomness
import numpy as np
import random


@TRANSFORMS.register_module()
class DiceGaussianNoise(BaseTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit: variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean: mean of the noise. Default: 0
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
            self,
            var_limit: Union[int, float, Tuple[float, float]] = (10.0, 50.0),
            mean: float = 0,
            prob: float = 0.5
    ) -> None:
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                msg = "Lower var_limit should be non negative."
                raise ValueError(msg)
            if var_limit[1] < 0:
                msg = "Upper var_limit should be non negative."
                raise ValueError(msg)
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                msg = "var_limit should be non negative."
                raise ValueError(msg)

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(f"Expected var_limit type to be one of (int, float, tuple, list), got {type(var_limit)}")

        self.mean = mean
        self.p = prob

    @cache_randomness
    def _make_noise(self, image):
        p = np.random.rand()
        if p <= self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            gauss = np.random.normal(self.mean, sigma, image.shape)
        else:
            gauss = 0
        return gauss

    def transform(self, results: dict) -> dict:

        image = results['img']
        gauss = self._make_noise(image)
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)
        results['img'] = noisy_image

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(var_limit={self.var_limit}, '
        repr_str += f'mean={self.mean}, '
        repr_str += f'p={self.p}'
        return repr_str
