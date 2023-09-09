import optuna
from optuna.trial import TrialState

from torchfm.torch_utils.constants import save_optuna_results_file


def get_study_stats(study_obj, mdl_name):
    assert study_obj is not None
    pruned_trials = study_obj.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study_obj.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    trial = study_obj.best_trial

    str_builder = "\n"
    str_builder += "Study statistics: "
    str_builder += f"  Study name: {study_obj.study_name} model name: {mdl_name}\n"
    str_builder += f"  Number of finished trials:  {len(study_obj.trials)}\n"
    str_builder += f"  Number of pruned trials:  {len(pruned_trials)}\n"
    str_builder += f"  Number of complete trials:  {len(complete_trials)}\n"

    str_builder += f"Best trial:\n"
    str_builder += f"  Value:  {trial.value}\n"
    str_builder += f"  Params: \n"
    for key, value in trial.params.items():
        str_builder += f"    {key}: {value}\n"

    return str_builder


def save_to_file(study, model_name, best_params, best_trial):
    study_stats = get_study_stats(study, model_name)
    with open(save_optuna_results_file, 'a+') as f:
        f.write(study_stats)
        f.write(str(best_params) + "\n" + str(best_trial) + "\n\n")
