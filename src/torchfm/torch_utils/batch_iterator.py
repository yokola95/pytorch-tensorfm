import torch
import numpy as np


def batch_iter_dataset(dataset, device, batch_size=2048, shuffle=True):
    """
    dataset: features and labels dataset
    """
    size_data = len(dataset)
    items_tensor = torch.tensor(dataset.items)  #, device=device)       # feature values
    targets_tensor = torch.tensor(dataset.targets)  #, device=device)   # labels

    if shuffle:
        idxs = torch.randperm(size_data)  #, device=device)
    else:
        idxs = torch.arange(size_data)  # , device=device)

    for batch_idxs in idxs.split(batch_size):
        yield items_tensor[batch_idxs], targets_tensor[batch_idxs]
