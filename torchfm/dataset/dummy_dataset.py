import numpy as np
import pandas as pd
import torch.utils.data


class DummyDataset(torch.utils.data.Dataset):
    """
    Dummy Dataset

    Data preparation
        splits the dataset to the features and the label

    :param dataset_path: dataset path
    """

    # path = "./torchfm//test-datasets/dummy.txt"
    path = "../test-datasets/dummy.txt"
    label = 'label'

    def __init__(self, dataset_path=path, sep=',', engine='c', header=None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)

        self.items = data.loc[:, data.columns != self.label].to_numpy()
        self.targets = data.loc[:, self.label].to_numpy()
        self.field_dims = np.max(self.items, axis=0)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
