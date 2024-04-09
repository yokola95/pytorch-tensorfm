import torch
import numpy as np


class BatchIter:
    def __init__(self, dataset, device, batch_size=1024, shuffle=True):
        self.multivalued = dataset.multivalued
        self.size_data = len(dataset)
        self.device = device
        self.batch_size = batch_size
        self.items_tensors = self.get_items_tensor(dataset, device)  # Call the method to initialize self.items_tensor
        self.targets_tensor = torch.tensor(dataset.targets, device=device)  # labels
        self.shuffle = shuffle

    # Returns tuple of tensors as the items
    def get_items_tensor(self, dataset, device):
        if not self.multivalued:
            return torch.tensor(dataset.items, device=device),  # feature values
        else:
            return (torch.tensor(dataset.indices, device=device),
                    torch.tensor(dataset.offsets, device=device),
                    torch.tensor(dataset.weights, device=device))

    def __len__(self):
        return self.size_data

    def __iter__(self):
        if self.shuffle:
            # should be deterministic if torch.manual_seed(...) is set
            idxs = torch.randperm(self.size_data, device=self.device)
        else:
            idxs = torch.arange(self.size_data, device=self.device)

        idxs = idxs.split(self.batch_size)

        for batch_idxs in idxs:
            tmp_items_tensors = tuple((x[batch_idxs, ...] for x in self.items_tensors))
            tmp_targets_tensor = self.targets_tensor[batch_idxs]
            yield tmp_items_tensors, tmp_targets_tensor

