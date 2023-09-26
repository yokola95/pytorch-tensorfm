import numpy as np
import pandas as pd
import torch.utils.data


class WrapperMultivaluedDataset(torch.utils.data.Dataset):
    """
    General Multivalued Dataset

    Data preparation
        splits the dataset to the features and the label

    :param dataset_path: dataset path
    """
    multivalued = True

    # dataset_path = "/Users/viderman/Documents/workspace/factorization_machine_git/pytorch-fm/torchfm/test-datasets/movielens/ml-1m/train.csv"
    def __init__(self, dataset_path, sep=',', secondary_sep='|', engine='c', header='infer'):
        miltival_col_name = 'genres'
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)
        self.targets = data['label'].to_numpy()
        data = data.drop('label', axis=1)

        data[miltival_col_name] = data[miltival_col_name].str.split(secondary_sep).apply(lambda x: [int(i) for i in x])

        self.max_length = max(data[miltival_col_name].map(len))

        num_columns = len(data.columns)
        global_offsets = list(range(num_columns - 1))

        self.weights = data[miltival_col_name].apply(lambda x: WrapperMultivaluedDataset._create_weights(x, self.max_length, num_columns - 1)).to_numpy()
        self.offsets = data[miltival_col_name].apply(lambda x: WrapperMultivaluedDataset._create_offsets(x, num_columns - 2, global_offsets)).to_numpy()

        data[miltival_col_name] = data[miltival_col_name].apply(lambda x: x + [0] * (self.max_length - len(x)))

        lst_genres = ['genre' + str(i) for i in range(0, 6)]
        df_genres = pd.DataFrame(data[miltival_col_name].to_list(), columns=lst_genres)

        self.indices = pd.concat([data.drop([miltival_col_name], axis=1), df_genres], axis=1, join='inner').to_numpy()
        self.field_dims = np.max(self.indices)
        self.num_columns = num_columns

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return np.concatenate((self.indices[index], self.offsets[index], self.weights[index]), axis=0, dtype='float32'), self.targets[index]

    @staticmethod
    def _create_weights(lst, full_length, num_cols, align_val=0.0):
        length = len(lst)
        weights =  [] if length == 0 else [1.0] * num_cols + [1.0 / length] * length + [align_val] * (full_length - length)
        return weights

    @staticmethod
    def _create_offsets(lst, num_cols, global_offsets): return global_offsets + [num_cols + len(lst)]


    # def __preprocess_target(self, target):
    #     target[target <= 3] = 0
    #     target[target > 3] = 1
    #     return target

#data[genres_lst'len_to_pad'] = int(max_length - data['length_genres'])
#data['to_pad'] = data['genres'].map(lambda x: [0] * (max_length-len(x))

#lst = [torch.tensor(item) for item in data['genres']]

#ds = WrapperMultivaluedDataset("/Users/viderman/Documents/workspace/factorization_machine_git/pytorch-fm/torchfm/test-datasets/movielens/ml-1m/train.csv")

