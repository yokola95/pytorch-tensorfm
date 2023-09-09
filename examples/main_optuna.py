import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from main import top_main_for_optuna_call
from torchfm.torch_utils.constants import save_optuna_results_file, optuna_journal_log
from torchfm.torch_utils.optuna_utils import save_to_file


def objective(trial, model_name, device_ind=0):
    lr = trial.suggest_float('lr', 1e-4, 0.1)
    opt_name = trial.suggest_categorical("opt_name", ["adagrad", "sgd"])  # ["adam", "sgd", "adagrad"]
    # batch_size = trial.suggest_int('batch_size', 100, 1000)

    return top_main_for_optuna_call(opt_name, lr, model_name, trial, device_ind)


def run_all_for_model(model_name, device_ind):
    storage = JournalStorage(JournalFileStorage(optuna_journal_log))
    study = optuna.create_study(study_name=(model_name + " Trial1"), storage=storage)
    study.optimize(lambda trial: objective(trial, model_name, device_ind), n_trials=1)

    p1, p2 = study.best_params, study.best_trial
    save_to_file(study, model_name, p1, p2)
    # allTrials = study.trials


#for model_name in ['fwfm', 'lowrank_fwfm']:
#    run_all_for_model(model_name, 0)


# sgd best lr=0.03  valid_err=0.4364890456199646, fwfm
# sgd best lr=0.03  valid_err=0.43644803762435913, lowrank_fwfm

# adagrad lr=0.01 valid_error=0.4593891501426697, lowrank_fwfm
# adagrad lr=0.01 valid_error=0.4633690416812897, fwfm

# adagrad  with value: 0.4432646877969047   'lr': 0.014894609341451013, 'opt_name': 'adagrad'  5 epochs
