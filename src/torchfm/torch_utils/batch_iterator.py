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

    # def batches(self):
    #     if self.shuffle:
    #         # should be deterministic if torch.manual_seed(...) is set
    #         idxs = torch.randperm(self.size_data, device=self.device)
    #     else:
    #         idxs = torch.arange(self.size_data, device=self.device)
    #
    #     for batch_idxs in idxs.split(self.batch_size):
    #         yield self.items_tensors[batch_idxs], self.targets_tensor[batch_idxs]

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

# class BatchIterMultiValued(BatchIter):
#     def __init__(self, dataset, device, batch_size=1024, shuffle=True):
#         super().__init__(dataset, device, batch_size, shuffle)
#
#     def get_items_tensor(self, dataset, device):
#         np_items = np.concatenate((dataset.indices, dataset.offsets, dataset.weights), axis=1)
#         return torch.tensor(np_items, device=device).float()  # assigned here


