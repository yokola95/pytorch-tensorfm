import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from torchfm.torch_utils.constants import lowrank_fwfm, fwfm, pruned_fwfm, logloss, auc

models_to_check = [fwfm, lowrank_fwfm, pruned_fwfm]
metrics_to_optimize = [logloss, auc]
criteo_ranks = [1, 2, 3, 4, 5]
emb_sizes = [4, 8]
device_inds = list(range(4)) * 2

all_options_for_studies = [('lowrank_fwfm', 'logloss', 1, 4), ('lowrank_fwfm', 'logloss', 1, 8), ('lowrank_fwfm', 'logloss', 2, 4), ('lowrank_fwfm', 'logloss', 2, 8), ('lowrank_fwfm', 'logloss', 3, 4), ('lowrank_fwfm', 'logloss', 3, 8), ('lowrank_fwfm', 'logloss', 4, 4), ('lowrank_fwfm', 'logloss', 4, 8), ('lowrank_fwfm', 'logloss', 5, 4), ('lowrank_fwfm', 'logloss', 5, 8), ('lowrank_fwfm', 'auc', 1, 4), ('lowrank_fwfm', 'auc', 1, 8), ('lowrank_fwfm', 'auc', 2, 4), ('lowrank_fwfm', 'auc', 2, 8), ('lowrank_fwfm', 'auc', 3, 4), ('lowrank_fwfm', 'auc', 3, 8), ('lowrank_fwfm', 'auc', 4, 4), ('lowrank_fwfm', 'auc', 4, 8), ('lowrank_fwfm', 'auc', 5, 4), ('lowrank_fwfm', 'auc', 5, 8), ('pruned_fwfm', 'logloss', 1, 4), ('pruned_fwfm', 'logloss', 1, 8), ('pruned_fwfm', 'logloss', 2, 4), ('pruned_fwfm', 'logloss', 2, 8), ('pruned_fwfm', 'logloss', 3, 4), ('pruned_fwfm', 'logloss', 3, 8), ('pruned_fwfm', 'logloss', 4, 4), ('pruned_fwfm', 'logloss', 4, 8), ('pruned_fwfm', 'logloss', 5, 4), ('pruned_fwfm', 'logloss', 5, 8), ('pruned_fwfm', 'auc', 1, 4), ('pruned_fwfm', 'auc', 1, 8), ('pruned_fwfm', 'auc', 2, 4), ('pruned_fwfm', 'auc', 2, 8), ('pruned_fwfm', 'auc', 3, 4), ('pruned_fwfm', 'auc', 3, 8), ('pruned_fwfm', 'auc', 4, 4), ('pruned_fwfm', 'auc', 4, 8), ('pruned_fwfm', 'auc', 5, 4), ('pruned_fwfm', 'auc', 5, 8), ('fwfm', 'logloss', 0, 4), ('fwfm', 'logloss', 0, 8), ('fwfm', 'auc', 0, 4), ('fwfm', 'auc', 0, 8)]
lst_michael = all_options_for_studies[0:20]
lst_oren = all_options_for_studies[20:28]
lst_ariel = all_options_for_studies[28:36]
lst_naama = all_options_for_studies[36:]

# 8 processes
# Use: 'tmux attach'   to run seession to run the python from
# cntrl B, D   --- to disconnect

# sys.path.append('/home/viderman/persistent_drive/pytorch-fm')
#export PYTHONPATH=/home/viderman/persistent_drive/pytorch-fm

if __name__ == '__main__':

    #all_options_for_studies = [(m_to_check, met_to_opt, rank, emb_size) for m_to_check in [lowrank_fwfm, pruned_fwfm] for met_to_opt in metrics_to_optimize for rank in criteo_ranks for emb_size in emb_sizes]
    #all_options_for_studies_fwfm = [(fwfm, met_to_opt, 0, emb_size) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
    #all_options_for_studies.extend(all_options_for_studies_fwfm)

    queue = mp.Queue(100)
    for tpl in lst_michael:
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, device_ind), daemon=True) for device_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")
