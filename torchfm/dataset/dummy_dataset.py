import numpy as np
import pandas as pd
import torch.utils.data


class DummyDataset(torch.utils.data.Dataset):
    """
    Dummy Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path
    """

    # path = "./torchfm//test-datasets/dummy.txt"
    path = "../test-datasets/dummy.txt"

    def __init__(self, dataset_path=path, sep=',', engine='c', header=None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()
        # print(data)
        self.items = data[:, 1:]
        self.targets = data[:, 0]
        self.field_dims = np.max(self.items, axis=0)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

