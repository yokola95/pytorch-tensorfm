import optuna
from optuna.trial import TrialState

from torchfm.torch_utils.constants import save_optuna_results_file, optuna_journal_log, sep


# def get_study_stats(study_obj, trial, mdl_name, metric_top_optimize):
#     assert study_obj is not None
#
#     header = "study name,trial_number,model name,metric_to_opt, #finish.trials,#prunned trials, #complete trials,best tr. val, params"
#
#     pruned_trials = study_obj.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study_obj.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
#     trial = study_obj.best_trial
#
#     str_builder = "\n"
#     str_builder += f"{study_obj.study_name}{sep}{trial.number}{mdl_name}{sep}{metric_top_optimize}{sep}"
#     str_builder += f"{len(study_obj.trials)}{sep}"
#     str_builder += f"{len(pruned_trials)}{sep}"
#     str_builder += f"{len(complete_trials)}{sep}"
#     str_builder += f"{trial.value}{sep}"
#     for key, value in trial.params.items():
#         str_builder += f"{key}{sep}{value}{sep}"
#
#     return str_builder


def save_all_args_to_file(*args):
    str_args = [str(arg) for arg in args]
    with open(save_optuna_results_file, 'a+') as f:
        str_to_write = sep.join(str_args) + "\n"
        f.write(str_to_write)


# def save_to_file(study, model_name, metric_top_optimize, best_params, best_trial):
#     study_stats = get_study_stats(study, best_trial, model_name, metric_top_optimize)
#     save_all_args_to_file(study_stats)
#     # f.write(str(best_params) + "\n" + str(best_trial) + "\n\n")


def create_journal_name(base_journal_name, suffix):
    return base_journal_name + suffix + '.log'


def get_journal_name(model_name, metric_top_optimize, top_k_rank, emb_size):
    return f"{optuna_journal_log}_{model_name}_{metric_top_optimize}_{top_k_rank}_{emb_size}.log"


def erase_content_journal(journal_log):
    file_to_delete = open(journal_log, 'w+')
    file_to_delete.close()
