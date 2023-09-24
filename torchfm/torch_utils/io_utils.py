import os
from torchfm.torch_utils.constants import *


def get_train_validation_test_preprocessed_paths(base_path, base_filename):
    if 'movielens' in base_path:
        return [os.path.join(base_path, file_type + csv) for file_type in [train, validation, test]]

    return [os.path.join(base_path, base_filename + '_' + file_type + '_' + preprocessed + txt) for file_type in [train, validation, test]]


def get_train_validation_test_paths(base_path, base_filename):
    return [os.path.join(base_path, base_filename + '_' + file_type + txt) for file_type in [train, validation, test]]
