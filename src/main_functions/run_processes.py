import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from src.torchfm.torch_utils.constants import lowrank_fwfm, fwfm, pruned_fwfm, fm, tensorfm, mse, logloss, emb_sizes, metrics_to_optimize, \
    ranks_to_check, device_inds


# def generate_all_criteo_avazu_options():
#     all_options_for_studies = [(m_to_check, met_to_opt, rank, emb_size) for m_to_check in [lowrank_fwfm, pruned_fwfm]
#                                for met_to_opt in metrics_to_optimize for rank in ranks_to_check for emb_size in
#                                emb_sizes]
#     all_options_for_studies_fwfm = [(fwfm, met_to_opt, 0, emb_size) for met_to_opt in metrics_to_optimize for emb_size
#                                     in emb_sizes]
#     all_option_for_studies_fm = [(fm, met_to_opt, 0, emb_size) for met_to_opt in metrics_to_optimize for emb_size in
#                                  emb_sizes]
#     all_options_for_studies.extend(all_options_for_studies_fwfm)
#     all_options_for_studies.extend(all_option_for_studies_fm)
#     return all_options_for_studies
#
#
# # for Criteo and Avazu
# all_options_for_studies = [('lowrank_fwfm', 'logloss', 1, 8), ('lowrank_fwfm', 'logloss', 1, 16),
#                            ('lowrank_fwfm', 'logloss', 2, 8), ('lowrank_fwfm', 'logloss', 2, 16),
#                            ('lowrank_fwfm', 'logloss', 3, 8), ('lowrank_fwfm', 'logloss', 3, 16),
#                            ('lowrank_fwfm', 'logloss', 4, 8), ('lowrank_fwfm', 'logloss', 4, 16),
#                            ('lowrank_fwfm', 'logloss', 5, 8), ('lowrank_fwfm', 'logloss', 5, 16),
#                            ('lowrank_fwfm', 'auc', 1, 8), ('lowrank_fwfm', 'auc', 1, 16), ('lowrank_fwfm', 'auc', 2, 8),
#                            ('lowrank_fwfm', 'auc', 2, 16), ('lowrank_fwfm', 'auc', 3, 8),
#                            ('lowrank_fwfm', 'auc', 3, 16), ('lowrank_fwfm', 'auc', 4, 8),
#                            ('lowrank_fwfm', 'auc', 4, 16), ('lowrank_fwfm', 'auc', 5, 8),
#                            ('lowrank_fwfm', 'auc', 5, 16), ('pruned_fwfm', 'logloss', 1, 8),
#                            ('pruned_fwfm', 'logloss', 1, 16), ('pruned_fwfm', 'logloss', 2, 8),
#                            ('pruned_fwfm', 'logloss', 2, 16), ('pruned_fwfm', 'logloss', 3, 8),
#                            ('pruned_fwfm', 'logloss', 3, 16), ('pruned_fwfm', 'logloss', 4, 8),
#                            ('pruned_fwfm', 'logloss', 4, 16), ('pruned_fwfm', 'logloss', 5, 8),
#                            ('pruned_fwfm', 'logloss', 5, 16), ('pruned_fwfm', 'auc', 1, 8),
#                            ('pruned_fwfm', 'auc', 1, 16), ('pruned_fwfm', 'auc', 2, 8), ('pruned_fwfm', 'auc', 2, 16),
#                            ('pruned_fwfm', 'auc', 3, 8), ('pruned_fwfm', 'auc', 3, 16), ('pruned_fwfm', 'auc', 4, 8),
#                            ('pruned_fwfm', 'auc', 4, 16), ('pruned_fwfm', 'auc', 5, 8), ('pruned_fwfm', 'auc', 5, 16),
#                            ('fwfm', 'logloss', 0, 8), ('fwfm', 'logloss', 0, 16), ('fwfm', 'auc', 0, 8),
#                            ('fwfm', 'auc', 0, 16), ('fm', 'logloss', 0, 8), ('fm', 'logloss', 0, 16),
#                            ('fm', 'auc', 0, 8), ('fm', 'auc', 0, 16)]
#
#
# # movielens
# def generate_movielens_options():
#     movielens_options = [(m_to_check, mse, rank, emb_size) for m_to_check in [lowrank_fwfm, pruned_fwfm] for rank in
#                          ranks_to_check for emb_size in emb_sizes]
#     movielens_options_fwfm_fm = [(m_to_check, mse, 0, emb_size) for m_to_check in [fwfm, fm] for emb_size in emb_sizes]
#     movielens_options.extend(movielens_options_fwfm_fm)
#     return movielens_options
#
#
# # for MovieLens
# movielens_options_studies = [('lowrank_fwfm', 'mse', 1, 8), ('lowrank_fwfm', 'mse', 1, 16),
#                              ('lowrank_fwfm', 'mse', 2, 8), ('lowrank_fwfm', 'mse', 2, 16),
#                              ('lowrank_fwfm', 'mse', 3, 8), ('lowrank_fwfm', 'mse', 3, 16),
#                              ('lowrank_fwfm', 'mse', 4, 8), ('lowrank_fwfm', 'mse', 4, 16),
#                              ('lowrank_fwfm', 'mse', 5, 8), ('lowrank_fwfm', 'mse', 5, 16),
#                              ('pruned_fwfm', 'mse', 1, 8), ('pruned_fwfm', 'mse', 1, 16), ('pruned_fwfm', 'mse', 2, 8),
#                              ('pruned_fwfm', 'mse', 2, 16), ('pruned_fwfm', 'mse', 3, 8), ('pruned_fwfm', 'mse', 3, 16),
#                              ('pruned_fwfm', 'mse', 4, 8), ('pruned_fwfm', 'mse', 4, 16), ('pruned_fwfm', 'mse', 5, 8),
#                              ('pruned_fwfm', 'mse', 5, 16), ('fwfm', 'mse', 0, 8), ('fwfm', 'mse', 0, 16),
#                              ('fm', 'mse', 0, 8), ('fm', 'mse', 0, 16)]


# Tmp list for Avazu dataset
lst_tmp = [(tensorfm, logloss, 0, 8, [2, 3], [2, 2])]

# device_inds: 1-8 processes

# sys.path.append('/home/viderman/persistent_drive/pytorch-fm/src')
# export PYTHONPATH=$PYTHONPATH:/home/viderman/persistent_drive/pytorch-fm/src

if __name__ == '__main__':

    queue = mp.Queue()
    for tpl in lst_tmp:
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, dev_ind), daemon=True) for dev_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")
