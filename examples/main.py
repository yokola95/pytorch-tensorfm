import tqdm
from torch.utils.data import DataLoader

from torchfm.torch_utils.constants import test_datasets_path, tmp_save_dir, wrapper, default_base_filename
from torchfm.torch_utils.io_utils import get_train_validation_test_preprocessed_paths
from torchfm.torch_utils.utils import *
import time
import optuna


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0.0
    #tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for fields, target in data_loader:   #enumerate(tk0):  i,
        fields, target = fields.to(device), target.float().to(device)
        y = model(fields)
        loss = criterion(y, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item()
        # if (i + 1) % log_interval == 0:
            # tk0.set_postfix(loss=total_loss / log_interval)
            # total_loss = 0.0


def test(model, data_loader, criterion, device):
    model.eval()
    targets, predicts = [], []
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.float().to(device)
            y = model(fields)
            targets.append(target)    # list of tensors
            predicts.append(y)        # list of tensors

    all_predicts = torch.cat(predicts)
    all_targets = torch.cat(targets)
    loss = criterion(all_predicts, all_targets).item()
    ctr_loss, half_loss = get_baselines_log_loss(all_targets)

    return loss, ctr_loss, half_loss


def main(dataset_name, dataset_paths, model_name, epoch, opt_name, learning_rate, batch_size, criterion, weight_decay, device, save_dir, trial=None):
    num_workers = 0
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, dataset_paths[0])
    valid_dataset = get_dataset(dataset_name, dataset_paths[1])
    test_dataset = get_dataset(dataset_name, dataset_paths[2])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model = get_model(model_name, train_dataset).to(device)
    criterion = get_criterion(criterion)
    optimizer = get_optimizer(opt_name, model.parameters(), learning_rate, weight_decay)  # torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #early_stopper = EarlyStopper(num_trials=epoch, save_path=f'{save_dir}/{model_name}.pt')

    for epoch_i in range(epoch):
        start = time.time()
        train(model, optimizer, train_data_loader, criterion, device)
        end = time.time()
        valid_err, _, _ = test(model, valid_data_loader, criterion, device)

        # Handle pruning based on the intermediate value.
        if trial is not None:
            trial.report(valid_err, epoch_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print_msg('epoch:', epoch_i, 'validation error:', valid_err, "train time:", end-start)
        #if not early_stopper.is_continuable(model, optimizer, valid_err):
        #    print(f'validation: best error: {early_stopper.best_error}')
        #    break
    test_err, ctr_err, half_err = test(model, test_data_loader, criterion, device)
    print_msg(f'test error: {test_err}, ctr error: {ctr_err}, 1/2 prediction error: {half_err}')
    return test_err


def top_main_for_optuna_call(opt_name, learning_rate, trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path, default_base_filename)
    err = main(wrapper, train_valid_test_paths, 'lowrank_fwfm', 20, opt_name, learning_rate, 100, 'bcelogitloss', 1e-6, device, tmp_save_dir, trial)
    return err


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path, default_base_filename)
#     main(wrapper, train_valid_test_paths, 'lowrank_fwfm', 20, "adagrad", 0.001, 100, 'bcelogitloss', 1e-6, device, tmp_save_dir)


    #from torchfm.torch_utils.parsing_datasets.criteo.criteo_parsing import CriteoParsing
    #CriteoParsing.do_action("split")


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
