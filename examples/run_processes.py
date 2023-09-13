import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from torchfm.torch_utils.constants import lowrank_fwfm, fwfm, pruned_fwfm, logloss, auc

models_to_check = [fwfm, lowrank_fwfm, pruned_fwfm]
metrics_to_optimize = [logloss, auc]
criteo_ranks = [1, 3, 5, 7, 9]
device_inds = range(1)  #range(4)

# sys.path.append('/home/viderman/persistent_drive/pytorch-fm')

if __name__ == '__main__':
    queue = mp.Queue(100)

    all_options_for_studies = [(m_to_check, met_to_opt, rank) for m_to_check in [lowrank_fwfm, pruned_fwfm] for met_to_opt in metrics_to_optimize for rank in criteo_ranks]  #models_to_check.zip(metric_top_optimize).zip(criteo_ranks):
    all_options_for_studies1 = [(fwfm, met_to_opt, 0) for met_to_opt in metrics_to_optimize]
    all_options_for_studies.extend(all_options_for_studies1)

    for tpl in all_options_for_studies:
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, device_ind), daemon=True) for device_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")

    #
    #
    # # processes = [Process(target=run_all_for_model, args=(m[0], m[1]), daemon=True) for m in models_to_check]
    #
