import os
import pandas as pd
import torch
import src
from src.torchfm.torch_utils.constants import *
from src.torchfm.torch_utils.options_to_run import Option2Run


def option_to_file_name(option: Option2Run):
    return f"results_{option.m_to_check}_{option.met_to_opt}_{option.opt_name}_{option.emb_size}_{option.rank}_{option.reg_coef_vectors}_{option.reg_coef_biases}"


def get_train_validation_test_preprocessed_paths(base_path, base_filename):
    if 'movielens' in base_path:
        return [os.path.join(base_path, file_type + csv) for file_type in [train, validation, test]]
    elif 'avazu' in base_path:
        return [os.path.join(base_path, "final_" + file_type + csv) for file_type in [train, val, test]]
    else:  # criteo
        return [os.path.join(base_path, base_filename + '_' + file_type + '_' + preprocessed + txt) for file_type in [train, validation, test]]


def get_train_validation_test_paths(base_path, base_filename):
    return [os.path.join(base_path, base_filename + '_' + file_type + txt) for file_type in [train, validation, test]]


def read_df_from_hdfs(dataset_path, sep, engine, header):
    import tensorflow_io
    import tensorflow as tf

    with tf.io.gfile.GFile(dataset_path, "r") as f:
        df = pd.read_csv(f, sep, engine, header)
    return df


def read_pd_dataframe(dataset_path, sep, engine, header):
    if hdfs_run:
        return read_df_from_hdfs(dataset_path, sep, engine, header)  # df = spark.read.option("delimiter", sep).option("inferSchema", True).option("header", True).csv(dataset_path)
    else:
        return pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)


def write_to_hdfs(sep, option_to_run, *args):
    import tensorflow_io
    import tensorflow as tf

    path_suffix = option_to_file_name(option_to_run)
    hdfs_path = save_run_results + path_suffix

    str_args = [str(arg) for arg in args]
    line_str = sep.join(str_args)

    with tf.io.gfile.GFile(hdfs_path, "w") as f:
        f.write(line_str)
    #append_to_file(spark, line_str, hdfs_path)
    # df = spark.createDataFrame(data=[line_str], schema=["all_cols"])
    # df.write.save(path='csv', format='csv', mode='append', sep='\t')


def write_to_file(*args, sep, file_path):
    str_args = [str(arg) for arg in args]
    with open(file_path, 'a+') as f:
        str_to_write = sep.join(str_args) + "\n"
        f.write(str_to_write)


def write_debug_info(*args):
    write_to_file(*args, sep='\n', file_path=debug_info_file)


def save_all_args_to_file(option_to_run, *args):
    if not hdfs_run:
        write_to_file(*args, sep=',', file_path=save_optuna_results_file)
    else:
        write_to_hdfs(sep=',', option_to_run=option_to_run, *args)


def load_model(model_name, dataset, path):
    model = src.torchfm.torch_utils.utils.get_model(model_name, dataset=dataset)

    model.load_state_dict(torch.load(f'{tmp_save_dir}/{model_name}.pt'))
    checkpoint = torch.load(path)
    epoch_num = checkpoint['epoch']
    learning_rate = checkpoint['lr']
    opt_name = checkpoint['opt_name']
    model.eval()
    return model


def save_model(model, model_name, epoch_num, optimizer, learning_rate, opt_name, loss):
    torch.save({'epoch': epoch_num, 'lr': learning_rate, 'opt_name': opt_name, 'loss': loss, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'{tmp_save_dir}/{model_name}.pt')


def save_tensor(x, model_name):
    torch.save(x, f'{tmp_save_dir}/{model_name}_interaction.pt')


def load_tensor(model_name):
    return torch.load(f'{tmp_save_dir}/{model_name}_interaction.pt')
