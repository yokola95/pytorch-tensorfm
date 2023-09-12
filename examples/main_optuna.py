import numpy as np
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from main import top_main_for_optuna_call
from torchfm.torch_utils.constants import optuna_num_trials, fwfm, lowrank_fwfm, pruned_fwfm, logloss
from torchfm.torch_utils.optuna_utils import save_to_file, get_journal_name, erase_content_journal
from torchfm.torch_utils.utils import get_from_queue


def objective(study, trial, model_name, device_ind=0, metric_to_optimize="logloss", top_k_rank=5):
    lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    opt_name = trial.suggest_categorical("opt_name", ["adagrad", "sgd"])  # ["adam", "sgd", "adagrad"]
    # batch_size = trial.suggest_int('batch_size', 100, 500)

    return top_main_for_optuna_call(opt_name, lr, model_name, study, trial, device_ind, metric_to_optimize, top_k_rank)


def run_all_for_model(queue, device_ind):  # (model_name, device_ind, top_k_rank):
    while True:
        elm = get_from_queue(queue)
        if elm is None:
            return

        model_name = elm[0]
        metric_top_optimize = elm[1]
        top_k_rank = elm[2]

        journal_name = get_journal_name(model_name)
        erase_content_journal(journal_name)
        storage = JournalStorage(JournalFileStorage(journal_name))
        study = optuna.create_study(study_name=("Study " + model_name), storage=storage)
        # metric_top_optimize = logloss
        study.optimize(lambda trial: objective(study, trial, model_name, device_ind, metric_top_optimize, top_k_rank), n_trials=optuna_num_trials)

        #p1, p2 = study.best_params, study.best_trial


#for m_name in [lowrank_fwfm, fwfm, pruned_fwfm]:
#    run_all_for_model(m_name, 0, 5)
