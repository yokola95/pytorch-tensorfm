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
val = "val"
test = "test"
fwfm = "fwfm"
fm = 'fm'
lowrank_fwfm = "lowrank_fwfm"
pruned_fwfm = "pruned_fwfm"
logloss = "logloss"
auc = "auc"
mse = "mse"
sep = ","
minimize = "minimize"
maximize = "maximize"
reg_param = "reg_param"

debug_print = False
sparseGrads = True
epochs_num = 10
# batch_size = 256
# top_k_percent = 0.05
weight_decay = 0
use_regularization = False

base_path_project = "hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/viderman/low_rank_experiments/data"
path_torchfm = "/pytorch-fm/src/torchfm".format(base_path_project)
tmp_save_dir = '{}/tmp_save_dir'.format(base_path_project)

test_datasets_path_movielens = "{}/test-datasets/movielens/ml-1m".format(base_path_project)
test_datasets_path_avazu = "{}/test-datasets/avazu".format(base_path_project)
test_datasets_path_criteo = "{}/test-datasets/criteo".format(base_path_project)
test_datasets_path = test_datasets_path_criteo

default_base_filename = 'train1M'

original_input_file_path = '{}/{}.txt'.format(test_datasets_path, default_base_filename)

optuna_num_trials = 10
debug_info_file = f"{tmp_save_dir}/debug_info.txt"
save_optuna_results_file = f"{tmp_save_dir}/optuna_results.txt"
save_run_results = f"{tmp_save_dir}/run_results/"
optuna_journal_log = f"{tmp_save_dir}/optuna-journal"

optuna_journal_log_fwfm = f"{tmp_save_dir}/optuna-journal_fwfm"
optuna_journal_log_lr_fwfm = f"{tmp_save_dir}/optuna-journal_lr_fwfm"
optuna_journal_log_pruned_fwfm = f"{tmp_save_dir}/optuna_journal_log_pruned_fwfm"

torch_global_seed = 0
python_random_seed = 0

hdfs_run = True

