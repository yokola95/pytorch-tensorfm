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
val = "val"
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

criteo = "criteo"
avazu = "avazu"
movielens = "movielens"

debug_print = False
sparseGrads = True
epochs_num = 20
batch_sizes_to_check = [1024]
emb_sizes = [8, 16]
# top_k_percent = 0.05
weight_decay = 0
coef_vectors_max = 1e-4
coef_vectors_min = 0.0
coef_biases_max = 1e-4
coef_biases_min = 0.0

models_to_check = [fwfm, lowrank_fwfm, pruned_fwfm]
metrics_to_optimize = [logloss, auc]
ranks_to_check = [1, 2, 3, 4, 5]
device_inds = list(range(4))

base_path_project="/Users/<user_name>/Documents/workspace/factorization_machine_git/pytorch-fm/data"
tmp_save_dir = '{}/tmp_save_dir'.format(base_path_project)

dataset_name = movielens
test_datasets_path = "{}/test-datasets/{}".format(base_path_project,dataset_name)

optuna_num_trials = 50
debug_info_file = f"{tmp_save_dir}/debug_info.txt"
save_optuna_results_file = f"{tmp_save_dir}/optuna_results.txt"
save_run_results = f"{tmp_save_dir}/run_results/"
optuna_journal_log = f"{tmp_save_dir}/optuna-journal"

optuna_journal_log_fwfm = f"{tmp_save_dir}/optuna-journal_fwfm"
optuna_journal_log_lr_fwfm = f"{tmp_save_dir}/optuna-journal_lr_fwfm"
optuna_journal_log_pruned_fwfm = f"{tmp_save_dir}/optuna_journal_log_pruned_fwfm"

torch_global_seed = 42
python_random_seed = 42
optuna_seed = 42
use_batch_iterator = True

hdfs_run = False

# preprocessing constants
default_base_filename = 'train1M'  #'train'
original_input_file_path = '{}/{}.txt'.format(test_datasets_path, default_base_filename)
