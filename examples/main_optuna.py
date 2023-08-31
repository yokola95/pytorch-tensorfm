import optuna
from main import top_main_for_optuna_call


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 0.1)
    opt_name = trial.suggest_categorical("opt_name", ["adam", "sgd", "adagrad"])
    # batch_size = trial.suggest_uniform('batch_size', 100, 10000)

    return top_main_for_optuna_call(opt_name, lr, trial)


study = optuna.create_study(study_name="Trial")
study.optimize(objective, n_trials=10)


p = study.best_params, study.best_trial
print(p)

allTrials = study.trials
