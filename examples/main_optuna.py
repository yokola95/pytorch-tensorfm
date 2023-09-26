import traceback
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from main import top_main_for_optuna_call
from torchfm.torch_utils.constants import optuna_num_trials, logloss, mse, minimize, maximize
from torchfm.torch_utils.optuna_utils import get_journal_name, erase_content_journal
from torchfm.torch_utils.utils import get_from_queue, write_debug_info


def objective(study, trial, model_name, device_ind, metric_to_optimize, rank_param, emb_size):
    lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    opt_name = "adagrad"  #  trial.suggest_categorical("opt_name", ["adagrad"])  # , "sgd" # ["adam", "sparseadam"]  make issues with sparse/dense gradients
    batch_size = 256  #trial.suggest_int('batch_size', 100, 1000, log=True)

    return top_main_for_optuna_call(opt_name, lr, model_name, study, trial, device_ind, metric_to_optimize, rank_param, batch_size, emb_size)


def run_optuna_study(model_name, metric_to_optimize, rank_param, emb_size, device_ind):

    journal_name = get_journal_name(model_name, metric_to_optimize, rank_param, emb_size)
    erase_content_journal(journal_name)
    storage = JournalStorage(JournalFileStorage(journal_name))
    study = optuna.create_study(study_name=f"Study {model_name} {metric_to_optimize} {rank_param} {emb_size}", storage=storage,
                                direction=(minimize if metric_to_optimize in [logloss, mse] else maximize))

    study.optimize(lambda trial: objective(study, trial, model_name, device_ind, metric_to_optimize, rank_param, emb_size), n_trials=optuna_num_trials)


def run_all_for_device_ind(queue, device_ind):
    try:
        while True:
            elm = get_from_queue(queue)
            if elm is None:
                write_debug_info("run_all_for_device_ind no elm returned. Exit. Device ind: ", str(device_ind))
                return

            model_name, metric_top_optimize, rank_param, emb_size = elm[0], elm[1], elm[2], elm[3]
            run_optuna_study(model_name, metric_top_optimize, rank_param, emb_size, device_ind)
    except Exception as e:
        write_debug_info("Exception in run_all_for_device_ind: ", str(e), traceback.format_exc())



#for m_name in ["pruned_fwfm"]:  # lowrank_fwfm, fwfm,
#    run_optuna_study(m_name, logloss, 2, 4, 0)
