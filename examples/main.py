import torch
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from torchfm.torch_utils.utils import *
import time


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        target = target.float()
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return mean_squared_error(targets, predicts)  # roc_auc_score


def main(dataset_name,
         dataset_path,
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
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model = get_model(model_name, dataset).to(device)
    criterion = get_criterion(criterion)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=epoch, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        start = time.time()
        train(model, optimizer, train_data_loader, criterion, device)
        end = time.time()
        mse = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: mse:', mse, "train time:", end-start)
        if not early_stopper.is_continuable(model, optimizer, mse):
            print(f'validation: best error: {early_stopper.best_error}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test error: {mse}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main('dummy', '../torchfm/test-datasets/dummy.txt', 'lowrank_fwfm', 5, 0.01, 10, 'bcelogitloss', 1e-6, device, '../tmp_save_dir')


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
