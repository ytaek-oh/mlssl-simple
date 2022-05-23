import itertools
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def split_indices(data_size, percent, seed):
    rng = np.random.RandomState(seed)
    data_inds = list(range(data_size))
    rng.shuffle(data_inds)

    split_size = int(data_size * percent / 100)
    return data_inds[:split_size], data_inds[split_size:]


class Subset:

    def __init__(self, dataset, indices, transforms=None):
        self.dataset = deepcopy(dataset)
        self.indices = indices
        if transforms is not None:
            self.dataset.transforms = transforms

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class InfiniteSampler(Sampler):

    def __init__(self, size: int, shuffle: bool = True, seed: int = 1):
        self._size = size
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size).tolist()
            else:
                yield from torch.arange(self._size).tolist()

    def __len__(self):
        return self._size
