import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler


class DistributedWeightedSampler(DistributedSampler, WeightedRandomSampler):
    def __init__(self, dataset, num_replicas, rank, replacement=True):
        DistributedSampler.__init__(self, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        num_samples = self.total_size
        self.weights = self._get_weights(dataset)
        WeightedRandomSampler.__init__(self, self.weights, num_samples, replacement=replacement)

    def __iter__(self):
        indices = list(WeightedRandomSampler.__iter__(self))
        return iter(indices[self.rank:self.total_size: self.num_replicas])

    def _get_weights(self, dataset):
        samples_weight = torch.from_numpy(dataset.dataset.compute_sample_weight(dataset.indices))
        return samples_weight
