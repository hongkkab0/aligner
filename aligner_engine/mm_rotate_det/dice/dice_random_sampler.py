import logging
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

logger = logging.getLogger(__name__)


@DATA_SAMPLERS.register_module()
class DiceRandomSubsetSampler(Sampler):
    """
    Args:
        shuffle (bool): whether to shuffle the indices or not
        subset_ratio (float): the ratio of subset data to sample from the underlying dataset
        seed (int): the initial seed of the shuffle. Must be the same
            across all workers. If None, will use a random seed shared
            among workers (require synchronization among all workers).
    """ 

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 subset_ratio: float = 0.3,
                 seed: Optional[int] = None) -> None:
        assert 0.0 < subset_ratio <= 1.0

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.num_samples = math.ceil(
            (len(self.dataset) - rank) / world_size)
        self.total_size = len(self.dataset)
        self._size_subset = int(self.total_size * subset_ratio)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        self._indexes_subset = indices[: self._size_subset]

        logger.info("Using DiceRandomSubsetSampler......")
        logger.info(f"Randomly sample {self._size_subset} data from the original {self.total_size} data")

        return iter(self._indexes_subset)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
