import optuna
from main import top_main_for_optuna_call
from torchfm.torch_utils.constants import save_optuna_results_file

model_name = "fwfm"


def save_to_file(model_name, best_params, best_trial):
    with open(save_optuna_results_file, 'a+') as f:
        f.write(model_name + "\n" + str(best_params) + "\n" + str(best_trial) + "\n\n")


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 0.1)
    opt_name = trial.suggest_categorical("opt_name", ["adagrad"])  # ["adam", "sgd", "adagrad"]
    # batch_size = trial.suggest_int('batch_size', 100, 1000)

    return top_main_for_optuna_call(opt_name, lr, model_name, trial, 0)


# lowrank_

study = optuna.create_study(study_name=(model_name + " Trial"), direction="minimize")
study.optimize(objective, n_trials=10)

p1, p2 = study.best_params, study.best_trial
print(p1)
print(p2)
save_to_file(model_name, p1, p2)

allTrials = study.trials

# sgd best lr=0.03  valid_err=0.4364890456199646, fwfm
# sgd best lr=0.03  valid_err=0.43644803762435913, lowrank_fwfm

# adagrad lr=0.01 valid_error=0.4593891501426697, lowrank_fwfm
# adagrad lr=0.01 valid_error=0.4633690416812897, fwfm
