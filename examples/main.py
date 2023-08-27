import torch
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss
from torch.utils.data import DataLoader
import numpy as np

from torchfm.torch_utils.constants import test_datasets_path, tmp_save_dir, wrapper, default_base_filename
from torchfm.torch_utils.io_utils import get_train_validation_test_preprocessed_paths
from torchfm.torch_utils.utils import *
import time


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    ctr_loss_f = torch.nn.BCELoss()
    model.train()
    total_loss = 0
    total_ctr_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        target = target.float()
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()

        target_ctr = np.sum(target.float().tolist())/len(target.tolist())
        total_ctr_loss += ctr_loss_f(torch.tensor(np.full(target.size(dim=0), target_ctr)).float(), target.float()).item()    # global train ctr 0.22711533894173677

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval, ctr_loss=total_ctr_loss / log_interval)
            total_loss = 0
            total_ctr_loss = 0


def test(model, data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return criterion(torch.tensor(predicts), torch.tensor(targets).float()).item()      #log_loss(targets, sigmoid(predicts))  # roc_auc_score


def main(dataset_name,
         dataset_paths,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         criterion,
         weight_decay,
         device,
         save_dir):
    num_workers = 1
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, dataset_paths[0])
    valid_dataset = get_dataset(dataset_name, dataset_paths[1])
    test_dataset = get_dataset(dataset_name, dataset_paths[2])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model = get_model(model_name, train_dataset).to(device)
    criterion = get_criterion(criterion)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=epoch, save_path=f'{save_dir}/{model_name}.pt')

    for epoch_i in range(epoch):
        start = time.time()
        train(model, optimizer, train_data_loader, criterion, device)
        end = time.time()
        err = test(model, valid_data_loader, criterion, device)
        print('epoch:', epoch_i, 'validation error:', err, "train time:", end-start)
        if not early_stopper.is_continuable(model, optimizer, err):
            print(f'validation: best error: {early_stopper.best_error}')
            break
    test_err = test(model, test_data_loader, criterion, device)
    print(f'test error: {test_err}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path, default_base_filename)
    main(wrapper, train_valid_test_paths, 'fwfm', 20, 0.001, 100, 'bcelogitloss', 1e-6, device, tmp_save_dir)

# lowrank_fwfm
    #from torchfm.torch_utils.parsing_datasets.criteo.criteo_parsing import CriteoParsing
    #CriteoParsing.do_action("transform")


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_name', default='criteo')
#     parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
#     parser.add_argument('--model_name', default='afi')
#     parser.add_argument('--epoch', type=int, default=100)
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     parser.add_argument('--batch_size', type=int, default=2048)
#     parser.add_argument('--weight_decay', type=float, default=1e-6)
#     parser.add_argument('--device', default='cuda:0')
#     parser.add_argument('--save_dir', default='chkpt')
#     args = parser.parse_args()
#     main(args.dataset_name,
#          args.dataset_path,
#          args.model_name,
#          args.epoch,
#          args.learning_rate,
#          args.batch_size,
#          args.weight_decay,
#          args.device,
#          args.save_dir)
