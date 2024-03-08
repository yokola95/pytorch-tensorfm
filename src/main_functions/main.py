from torchmetrics.classification import auroc

from src.torchfm.torch_utils.constants import epochs_num, test_datasets_path, wrapper, auc, movielens, mse, dataset_name
from src.torchfm.torch_utils.io_utils import get_train_validation_test_preprocessed_paths, save_all_args_to_file
from src.torchfm.torch_utils.optuna_utils import prune_running_if_needed
import torch
import time

from src.torchfm.torch_utils.utils import set_torch_seed, get_device_str, get_iterators, get_datasets, BestError, \
    get_criterion, get_optimizer, EarlyStopper, get_model, print_msg


def train(model, optimizer, iterator, criterion, device, option_to_run):
    model.train()

    for fields, target in iterator.batches():
        fields, target = fields.to(device), target.float().to(device)
        y, reg = model(fields, option_to_run.return_l2)    # return y,regularization_term_arr
        loss = criterion(y, target)
        total_reg = reg[0] * option_to_run.reg_coef_vectors + reg[1] * option_to_run.reg_coef_biases
        cost = loss + total_reg
        model.zero_grad()           # optimizer.zero_grad()
        cost.backward()
        optimizer.step()


def train_wrapper(model, optimizer, iterator, criterion, device, option_to_run):
    start = time.time()
    train(model, optimizer, iterator, criterion, device, option_to_run)
    end = time.time()
    return end - start


def test(model, iterator, criterion, device):
    model.eval()
    test_loss_sum = torch.zeros(1, device=device)
    test_set_size = 0

    ys = []
    targets = []
    auc = auroc.AUROC(task="binary")  # BinaryAUROC()

    with torch.no_grad():
        for fields, target in iterator.batches():   # tqdm.tqdm(..., smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.float().to(device)
            y, _ = model(fields)
            test_loss_sum += criterion(y, target) * target.shape[0]
            test_set_size += target.shape[0]
            ys.append(y)
            targets.append(target.type(torch.IntTensor))
            #auc.update(y, target)

    loss = test_loss_sum.item() / test_set_size
    # ctr_loss, half_loss = get_baselines_log_loss(all_targets) # compute this with sums
    auc_res = auc(torch.cat(ys).to(device), torch.cat(targets).to(device)).item()  # auc.compute().item()

    return loss, auc_res   # loss can be logloss or mse


def valid_test(model, valid_iterator, test_iterator, criterion, device):
    start = time.time()
    valid_err, valid_auc = test(model, valid_iterator, criterion, device)  # , ctr_err, half_err
    end = time.time()
    test_err, test_auc = test(model, test_iterator, criterion, device)
    return valid_err, valid_auc, test_err, test_auc, end - start


def main(dataset_nm, dataset_paths, option_to_run, epoch, criterion_name, weight_decay, device, study=None, trial=None):
    num_workers = 0
    device = torch.device(device)
    study_name = study.study_name if study is not None else ""
    trial_number = trial.number if trial is not None else 0

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_nm, dataset_paths)
    train_iterator, valid_iterator, test_iterator = get_iterators(train_dataset, valid_dataset, test_dataset, option_to_run.batch_size, num_workers, device)

    model = get_model(option_to_run.m_to_check, train_dataset, option_to_run.rank, option_to_run.emb_size).to(device)
    criterion = get_criterion(criterion_name)
    optimizer = get_optimizer(option_to_run.opt_name, model.parameters(), option_to_run.lr, weight_decay)
    early_stopper = EarlyStopper()
    best_error = BestError()

    for epoch_i in range(epoch):
        train_time = train_wrapper(model, optimizer, train_iterator, criterion, device, option_to_run)

        valid_err, valid_auc, test_err, test_auc, valid_time = valid_test(model, valid_iterator, test_iterator, criterion, device)
        save_all_args_to_file(option_to_run, study_name, option_to_run.to_csv(), trial_number, epoch_i, valid_err, valid_auc, test_err, test_auc, train_time, valid_time, criterion_name, dataset_nm)
        best_error.update(valid_err, valid_auc)

        prune_running_if_needed(trial, valid_err, epoch_i)
        early_stopper(valid_err)
        if early_stopper.early_stop:
            print(f'early stopped validation: best error: {early_stopper.best_loss}')
            break

    valid_err, valid_auc, test_err, test_auc, _ = valid_test(model, valid_iterator, test_iterator, criterion, device)
    print_msg(f'valid error: {valid_err} test error: {test_err}')

    return best_error.best_logloss, best_error.best_auc


def top_main_for_option_run(study, trial, device_ind, option_to_run):
    set_torch_seed()
    device_str = get_device_str(device_ind)

    train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path)

    criterion_name = mse if movielens in train_valid_test_paths[0] else 'bcelogitloss'
    valid_err, valid_auc = main(dataset_name, train_valid_test_paths, option_to_run, epochs_num, criterion_name, 0, device_str, study, trial)
    return valid_err if option_to_run.met_to_opt != auc else valid_auc
