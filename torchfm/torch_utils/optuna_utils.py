import optuna
from optuna.trial import TrialState

from torchfm.torch_utils.constants import save_optuna_results_file, optuna_journal_log
from torchfm.torch_utils.utils import write_to_file


def save_all_args_to_file(*args):
    write_to_file(*args, sep=',', file_path=save_optuna_results_file)


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
