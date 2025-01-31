import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from src.torchfm.torch_utils.constants import dcn,afm, lowrank_fwfm, fwfm, pruned_fwfm, fm, lr, tensorfm, mse, emb_sizes, metrics_to_optimize, \
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


# # movielens
# def generate_movielens_options():
#     movielens_options = [(m_to_check, mse, rank, emb_size) for m_to_check in [lowrank_fwfm, pruned_fwfm] for rank in
#                          ranks_to_check for emb_size in emb_sizes]
#     movielens_options_fwfm_fm = [(m_to_check, mse, 0, emb_size) for m_to_check in [fwfm, fm] for emb_size in emb_sizes]
#     movielens_options.extend(movielens_options_fwfm_fm)
#     return movielens_options


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



# code/paper split to context and item fields like in low rank paper

# tensorfm_options = [([2],[4]), ([2],[2]), ([2,3],[4,4]), ([2,3],[2,2])]
# tensorfm for Avazu
tensorfm_options = [([2,3],[8,8]), ([2,3],[16,16]), ([2],[22])] # new params to try ([2],[22])


lst_tensorfm_options = [(tensorfm, met_to_opt, 0, emb_size, tensorfm_option[0], tensorfm_option[1]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes for tensorfm_option in tensorfm_options]


fwfm_options = [(fwfm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
fm_options = [(fm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
lr_options = [(lr, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
dcn_options = [(dcn, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
afm_options = [(afm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]


# 8 processes
# Use: 'tmux attach'   to run session to run the python from
# ctrl B, D   --- to disconnect

# sys.path.append('/home/viderman/persistent_drive/pytorch-fm/src')
# export PYTHONPATH=$PYTHONPATH:/home/viderman/persistent_drive/pytorch-fm/src

if __name__ == '__main__':

    queue = mp.Queue()
    for tpl in (lst_tensorfm_options+fwfm_options):
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, dev_ind), daemon=True) for dev_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")
