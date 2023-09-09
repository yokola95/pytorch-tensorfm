import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import accuracy, auroc, f_beta

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.dataset.wrapper_dataset import WrapperDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.fwfm import FieldWeightedFactorizationMachineModel
from torchfm.model.low_rank_fwfm import LowRankFieldWeightedFactorizationMachineModel
from torchfm.torch_utils.constants import *


def print_msg(*args):
    if debug_print:
        print(args)


def get_optimizer(opt_name, parameters, learning_rate, weight_decay=0):
    if opt_name == "adam":
        return torch.optim.Adam(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "sparseadam":
        return torch.optim.SparseAdam(params=parameters, lr=learning_rate)
    elif opt_name == "adagrad":
        return torch.optim.Adagrad(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('unknown optimizer name: ' + opt_name)


def get_criterion(criterion):
    if criterion == 'bceloss':
        return torch.nn.BCELoss()
    elif criterion == 'bcelogitloss':
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == 'mseloss':
        return torch.nn.MSELoss()
    elif criterion == "nllloss":
        return torch.nn.NLLLoss()
    else:
        raise ValueError('unknown criterion name: ' + criterion)


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'wrapper':
        return WrapperDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_datasets(dataset_name, dataset_paths):
    train_dataset = get_dataset(dataset_name, dataset_paths[0])
    valid_dataset = get_dataset(dataset_name, dataset_paths[1])
    test_dataset = get_dataset(dataset_name, dataset_paths[2])
    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_data_loader, valid_data_loader, test_data_loader


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    num_features = dataset.field_dims
    features, labels = iter(dataset).__next__()
    num_columns = len(features)

    if name == 'lr':
        return LogisticRegressionModel(num_features)
    elif name == 'fm':
        return FactorizationMachineModel(num_features, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(num_features, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(num_features, embed_dim=4)
    elif name == 'fwfm':
        return FieldWeightedFactorizationMachineModel(num_features=num_features, embed_dim=4, num_fields=num_columns, topk=round(top_k_percent * num_columns))
    elif name == 'lowrank_fwfm':
        return LowRankFieldWeightedFactorizationMachineModel(num_features=num_features, embed_dim=4, num_fields=num_columns, c=round(top_k_percent * num_columns))
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(num_features, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(num_features, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(num_features, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            num_features, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(num_features, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             num_features, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            num_features, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def get_baselines_log_loss(targets):
    log_loss = torch.nn.BCELoss()
    targets_ctr = torch.sum(targets) / targets.size(dim=0)
    ctr_loss = log_loss(torch.ones_like(targets) * targets_ctr.item(), targets).item()  # global train ctr 0.22711533894173677
    half_loss = log_loss((torch.ones_like(targets) * 0.5).float(), targets).item()

    return ctr_loss, half_loss


def get_absolute_sizes(total_len, rel_sizes):
    assert np.sum(rel_sizes) == 1.0
    absolute_sizes = [round(rel * total_len) for rel in rel_sizes[:-1]]
    absolute_sizes.append(total_len - np.sum(absolute_sizes))
    assert total_len == np.sum(absolute_sizes)
    return absolute_sizes


def load_model(model_name, dataset, path):
    model = get_model(model_name, dataset=dataset)

    model.load_state_dict(torch.load(f'{tmp_save_dir}/{model_name}.pt'))
    checkpoint = torch.load(path)
    epoch_num = checkpoint['epoch']
    learning_rate = checkpoint['lr']
    opt_name = checkpoint['opt_name']
    model.eval()
    return model


def save_model(model, model_name, epoch_num, optimizer, learning_rate, opt_name, loss):
    torch.save({'epoch': epoch_num, 'lr': learning_rate, 'opt_name': opt_name, 'loss': loss, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'{tmp_save_dir}/{model_name}.pt')


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_error = None
        self.save_path = save_path

    def is_continuable(self, model, optimizer, error):
        if self.best_error is None or error < self.best_error:
            self.best_error = error
            self.trial_counter = 0
            # torch.save(model, self.save_path)
            save_model(model, self.trial_counter, optimizer, error)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class LossCalc:
    total_loss = 0
    total_ctr_loss = 0     # loss w.r.t. ctr prediction on data
    total_half_loss = 0    # loss w.r.t. constant 1/2 prediction

    def __init__(self, total_loss, total_ctr_loss, total_half_loss):
        self.total_loss = total_loss
        self.total_ctr_loss = total_ctr_loss
        self.total_half_loss = total_half_loss

    def add(self, total_loss, total_ctr_loss, total_half_loss):
        self.total_loss += total_loss
        self.total_ctr_loss += total_ctr_loss
        self.total_half_loss += total_half_loss

    def remove_results(self):
        self.total_loss = 0
        self.total_ctr_loss = 0
        self.total_half_loss = 0
