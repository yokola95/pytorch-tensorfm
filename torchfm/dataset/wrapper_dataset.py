import numpy as np
import pandas as pd
import torch.utils.data

from torchfm.torch_utils.constants import *


class WrapperDataset(torch.utils.data.Dataset):
    """
    Dummy Dataset

    Data preparation
        splits the dataset to the features and the label

    :param dataset_path: dataset path
    """

    # path = "./torchfm//test-datasets/dummy.txt"
    # path = "../test-datasets/dummy.txt"

    def __init__(self, dataset_path, sep='\t', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)

        self.items = data.loc[:, data.columns != label].to_numpy()
        self.targets = data.loc[:, label].to_numpy()
        self.field_dims = np.max(self.items)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
