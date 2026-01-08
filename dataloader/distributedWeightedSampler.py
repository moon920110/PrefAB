import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler


class DistributedWeightedSampler(DistributedSampler, WeightedRandomSampler):
    def __init__(self, dataset, num_replicas, rank, replacement=True):
        DistributedSampler.__init__(self, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.num_samples_local = self.num_samples

        total_samples = self.total_size
        self.weights = self._get_weights(dataset)

        WeightedRandomSampler.__init__(self, self.weights, total_samples, replacement=replacement)

    def __iter__(self):
        indices = list(WeightedRandomSampler.__iter__(self))
        return iter(indices[self.rank:self.total_size: self.num_replicas])

    def _get_weights(self, dataset):
        samples_weight = torch.from_numpy(dataset.dataset.compute_sample_weight(dataset.indices))
        return samples_weight

    def __len__(self):
        return self.num_samples_local


class WeightedSampler(WeightedRandomSampler):
    def __init__(self, dataset, replacement=True):
        num_samples = len(dataset)
        weights = self._get_weights(dataset)
        super().__init__(weights, num_samples, replacement=replacement)

    def _get_weights(self, dataset):
        samples_weight = torch.from_numpy(dataset.dataset.compute_sample_weight(dataset.indices))
        return samples_weight