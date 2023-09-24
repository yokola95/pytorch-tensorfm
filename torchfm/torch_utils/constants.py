unknown = 'unknown'
missing = 'missing'
label = 'label'
value2index_json = '.value2index.json'
index2value_json = '.index2value.json'
frequent_values_pkl = '.frequent_values.pkl'

wrapper = 'wrapper'
preprocessed = "preprocessed"
txt = ".txt"
csv = ".csv"
train = "train"
validation = "validation"
test = "test"
fwfm = "fwfm"
fm = 'fm'
lowrank_fwfm = "lowrank_fwfm"
pruned_fwfm = "pruned_fwfm"
logloss = "logloss"
auc = "auc"
sep = ","
minimize = "minimize"
maximize = "maximize"

debug_print = True
sparseGrads = True
epochs_num = 20
# batch_size = 256
# top_k_percent = 0.05
weight_decay = 0

base_path_project = "/Users/viderman/Documents/workspace/factorization_machine_git/pytorch-fm"
path_torchfm = "{}/torchfm".format(base_path_project)
tmp_save_dir = '{}/tmp_save_dir'.format(base_path_project)

test_datasets_path_movielens = "{}/test-datasets/movielens/ml-1m".format(path_torchfm)
test_datasets_path_avazu = "{}/test-datasets/avazu".format(path_torchfm)
test_datasets_path_criteo="{}/test-datasets/criteo".format(path_torchfm)
test_datasets_path = test_datasets_path_criteo

default_base_filename = 'train'

original_input_file_path = '{}/{}.txt'.format(test_datasets_path, default_base_filename)

optuna_num_trials = 10
debug_info_file = f"{tmp_save_dir}/debug_info.txt"
save_optuna_results_file = f"{tmp_save_dir}/optuna_results.txt"
optuna_journal_log = f"{tmp_save_dir}/optuna-journal"

optuna_journal_log_fwfm = f"{tmp_save_dir}/optuna-journal_fwfm"
optuna_journal_log_lr_fwfm = f"{tmp_save_dir}/optuna-journal_lr_fwfm"
optuna_journal_log_pruned_fwfm = f"{tmp_save_dir}/optuna_journal_log_pruned_fwfm"
