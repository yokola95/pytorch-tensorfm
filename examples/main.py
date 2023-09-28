from torchmetrics.classification import auroc
from torchfm.torch_utils.io_utils import get_train_validation_test_preprocessed_paths
from torchfm.torch_utils.optuna_utils import save_all_args_to_file
from torchfm.torch_utils.utils import *
import time
import optuna


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    # total_loss = 0.0
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


def train_wrapper(model, optimizer, data_loader, criterion, device, log_interval=100):
    start = time.time()
    train(model, optimizer, data_loader, criterion, device, log_interval=100)
    end = time.time()
    return end - start


def test(model, data_loader, criterion, device):
    model.eval()
    test_loss_sum = torch.zeros(1, device=device)
    test_set_size = 0

    auc = auroc.BinaryAUROC()

    with torch.no_grad():
        for fields, target in data_loader:   # tqdm.tqdm(..., smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.float().to(device)
            y = model(fields)
            test_loss_sum += criterion(y, target) * target.shape[0]
            test_set_size += target.shape[0]
            auc.update(y, target)

    loss = test_loss_sum.item() / test_set_size
    # ctr_loss, half_loss = get_baselines_log_loss(all_targets) # compute this with sums
    auc_res = auc.compute().item()

    return loss, auc_res   # loss can be logloss or mse


def valid_test(model, valid_data_loader, test_data_loader, criterion, device):
    start = time.time()
    valid_err, valid_auc = test(model, valid_data_loader, criterion, device)  # , ctr_err, half_err
    end = time.time()
    test_err, test_auc = test(model, test_data_loader, criterion, device)
    return valid_err, valid_auc, test_err, test_auc, end - start


def main(dataset_name, dataset_paths, model_name, epoch, opt_name, learning_rate, batch_size, emb_size, criterion_name, metric_to_optimize, weight_decay, device, rank_param, study=None, trial=None):
    num_workers = 0
    device = torch.device(device)
    study_name = study.study_name if study is not None else ""
    trial_number = trial.number if trial is not None else 0

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_name, dataset_paths)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers)

    model = get_model(model_name, train_dataset, rank_param, emb_size).to(device)
    criterion = get_criterion(criterion_name)
    optimizer = get_optimizer(opt_name, model.parameters(), learning_rate, weight_decay)
    early_stopper = EarlyStopper()
    best_error = BestError()

    for epoch_i in range(epoch):
        train_time = train_wrapper(model, optimizer, train_data_loader, criterion, device)

        valid_err, valid_auc, test_err, test_auc, valid_time = valid_test(model, valid_data_loader, test_data_loader, criterion, device)
        save_all_args_to_file(study_name, model_name, trial_number, epoch_i, valid_err, valid_auc, test_err, test_auc, train_time, valid_time, learning_rate, batch_size, emb_size, opt_name, criterion_name, metric_to_optimize, rank_param)
        best_error.update(valid_err, valid_auc)
        # Handle pruning based on the intermediate value.
        if trial is not None:
            trial.report(valid_err, epoch_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        early_stopper(valid_err)
        if early_stopper.early_stop:
            print(f'early stopped validation: best error: {early_stopper.best_loss}')
            break

    valid_err, valid_auc, test_err, test_auc, _ = valid_test(model, valid_data_loader, test_data_loader, criterion, device)
    print_msg(f'valid error: {valid_err} test error: {test_err}')

    # save_model(model, model_name + ' ' + str(trial_number) + ' ' + opt_name, epoch, criterion, learning_rate, opt_name, valid_err)

    return best_error.best_logloss, best_error.best_auc


def top_main_for_optuna_call(opt_name, learning_rate, model_name, study, trial, device_ind, metric_to_optimize, rank_param, batch_size, emb_size):
    device_str = ('cuda' if torch.cuda.is_available() else 'cpu') + ":" + str(device_ind)

    train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path, default_base_filename)
    dataset_name = 'movielens' if 'movielens' in train_valid_test_paths[0] else wrapper
    criterion_name = 'mse' if 'movielens' in train_valid_test_paths[0] else 'bcelogitloss'
    valid_err, valid_auc = main(dataset_name, train_valid_test_paths, model_name, epochs_num, opt_name, learning_rate, batch_size, emb_size, criterion_name, metric_to_optimize, 0, device_str, rank_param, study, trial)
    return valid_err if metric_to_optimize != auc else valid_auc
