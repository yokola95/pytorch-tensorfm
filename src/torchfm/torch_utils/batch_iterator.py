import torch
import numpy as np


#
#
# def batch_iter_dataset(dataset, device, batch_size=2048, shuffle=True):
#     """
#     dataset: features and labels dataset
#     """
#     size_data = len(dataset)
#     items_tensor = torch.tensor(dataset.items, device=device)       # feature values
#     targets_tensor = torch.tensor(dataset.targets, device=device)   # labels
#
#     if shuffle:
#         idxs = torch.randperm(size_data, device=device)
#     else:
#         idxs = torch.arange(size_data, device=device)
#
#     for batch_idxs in idxs.split(batch_size):
#         yield items_tensor[batch_idxs], targets_tensor[batch_idxs]


class BatchIter:
    def __init__(self, dataset, device, batch_size=1024, shuffle=True):
        self.size_data = len(dataset)
        self.device = device
        self.batch_size = batch_size
        self.items_tensor = self.get_items_tensor(dataset, device)  # Call the method to initialize self.items_tensor
        self.targets_tensor = torch.tensor(dataset.targets, device=device)  # labels
        self.shuffle = shuffle

    def get_items_tensor(self, dataset, device):
        return torch.tensor(dataset.items, device=device)  # feature values

    def batches(self):
        if self.shuffle:
            # should be deterministic if torch.manual_seed(...) is set
            idxs = torch.randperm(self.size_data, device=self.device)
        else:
            idxs = torch.arange(self.size_data, device=self.device)

        for batch_idxs in idxs.split(self.batch_size):
            yield self.items_tensor[batch_idxs], self.targets_tensor[batch_idxs]


class BatchIterMultiValued(BatchIter):
    def __init__(self, dataset, device, batch_size=1024, shuffle=True):
        super().__init__(dataset, device, batch_size, shuffle)

    def get_items_tensor(self, dataset, device):
        np_items = np.concatenate((dataset.indices, dataset.offsets, dataset.weights), axis=1)
        return torch.tensor(np_items, device=device).float()  # assigned here
