import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from torchfm.torch_utils.constants import lowrank_fwfm, fwfm, pruned_fwfm, logloss, auc

models_to_check = [fwfm, lowrank_fwfm, pruned_fwfm]
metrics_to_optimize = [logloss, auc]
criteo_ranks = [1, 2, 3, 4, 5]
emb_sizes = [4, 8]
device_inds = list(range(4)) * 2

# 8 processes
# Use: 'tmux attach'   to run seession to run the python from
# cntrl B, D   --- to disconnect

# sys.path.append('/home/viderman/persistent_drive/pytorch-fm')
#export PYTHONPATH=/home/viderman/persistent_drive/pytorch-fm

if __name__ == '__main__':
    queue = mp.Queue(100)

    all_options_for_studies = [(m_to_check, met_to_opt, rank, emb_size) for m_to_check in [lowrank_fwfm, pruned_fwfm] for met_to_opt in metrics_to_optimize for rank in criteo_ranks for emb_size in emb_sizes]
    all_options_for_studies_fwfm = [(fwfm, met_to_opt, 0, emb_size) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
    all_options_for_studies.extend(all_options_for_studies_fwfm)
    print(all_options_for_studies)

    for tpl in all_options_for_studies:
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, device_ind), daemon=True) for device_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")
