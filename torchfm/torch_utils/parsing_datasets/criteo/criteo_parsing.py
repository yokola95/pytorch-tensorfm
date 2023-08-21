import pickle
import json
import numpy as np
import pandas as pd
import torch
from math import floor, log
from pandas.api.types import is_numeric_dtype

print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available. Version:", torch.version.cuda)
else:
    print("CUDA is not available.")


# data_file = "../persistent_drive/data/train20K.txt"  # Replace with the actual path to your dataset file
unknown = 'unknown'
missing = 'missing'
label = 'label'
value2index_json = '.value2index.json'
index2value_json = '.index2value.json'
frequent_values_pkl = '.frequent_values.pkl'


class CriteoParsing:
    # Set the threshold (minimum frequency) for the default value
    threshold = 10

    # Default value to replace infrequent values
    default_value = 'other_'

    cat_names = [f'C{i}' for i in range(1, 27)]

    cont_names = [f'I{i}' for i in range(1, 14)]

    columns = [label, *(f'I{i}' for i in range(1, 14)), *(f'C{i}' for i in range(1, 27))]

    def __init__(self, base_path):
        self.data_file = base_path   # base path

    def read_dataset(self, data_file_path):
        df = pd.read_csv(data_file_path, sep='\t', names=self.columns)
        return df

    def save_dataset(self, df, save_file_path):
        df.to_csv(save_file_path, header=True, index=False)

    # Function to load frequent values from a file
    def load_frequent_values(self):
        file_path = self.data_file + frequent_values_pkl
        with open(file_path, 'rb') as f:
            frequent_values = pickle.load(f)
        return frequent_values

    def save_frequent_values(self, frequent_values):
        # save the frequent values to a file
        file_path = self.data_file + frequent_values_pkl
        with open(file_path, 'wb') as f:
            pickle.dump(frequent_values, f)

    # Function to store frequent values to a file
    def calc_save_frequent_values(self, df):
        frequent_values = {}
        for col in df.columns:
            if col.startswith('C'):
                # Calculate the frequency of each unique value in the column
                value_counts = df[col].value_counts()
                # Create a set of values that appear at least the threshold times
                frequent_values[col] = set(value_counts[value_counts >= self.threshold].index)

        self.save_frequent_values(self, frequent_values)

    # Function to replace infrequent values in each column with a default value
    def replace_infrequent_with_default(self, df, frequent_values):
        for col in df.columns:
            if col.startswith('C'):
                # Replace values with the default value based on the frequent values
                mask = (df[col].notnull()) & (~df[col].isin(frequent_values[col]))
                df.loc[mask, col] = self.default_value  # + col
                df[col].fillna('SP_NaN', inplace=True)

    def replace_numeric_with_bins(self, df):
        # Function to apply the custom transformation
        def numeric_transform(x):
            if np.isnan(x):
                return 'SP_NaN'
            elif x <= 1:
                return f'SP_{x:.0f}'
            else:
                return 'BN_' + str(floor(log(x) ** 2))

        # Filter numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop(label, errors='ignore')

        # Apply the custom transformation to the numerical columns
        df[numerical_cols] = df[numerical_cols].applymap(numeric_transform)

    # Save the index-value mapping to a file (debug)
    def save_index_value_mapping(self, global_index_value_mapping):
        with open(self.data_file + index2value_json, 'w') as mapping_file:
            json.dump(global_index_value_mapping, mapping_file)

    # Save the value-index mapping to a file (debug)
    def save_value_index_mapping(self, global_value_index_mapping):
        with open(self.data_file + value2index_json, 'w') as mapping_file:
            json.dump(global_value_index_mapping, mapping_file)

    # Load the index-value mapping to a file (debug)
    def load_index_value_mapping(self):
        with open(self.data_file + index2value_json, 'r') as mapping_file:
            global_index_value_mapping = json.load(mapping_file)
        return global_index_value_mapping

    # Save the value-index mapping to a file (debug)
    def load_value_index_mapping(self):
        with open(self.data_file + value2index_json, 'r') as mapping_file:
            global_value_index_mapping = json.load(mapping_file)
        return global_value_index_mapping

    def calc_index_value_mapping_from_column(self, column, offset):
        unique_values = column.unique()
        value_index_mapping = {unknown: offset, missing: offset + 1}
        value_index_mapping.update({value: offset + index + 2 for index, value in enumerate(unique_values)})

        return value_index_mapping

    def index_column_from_mapping(self, column, value_index_mapping):
        def map_val_to_ind(x):
            if x in value_index_mapping:
                return value_index_mapping[x]
            elif (is_numeric_dtype(column) and np.isnan(x)) or (x is None):
                return value_index_mapping[missing]
            else:
                return value_index_mapping[unknown]

        #new_column = column.map(value_index_mapping)
        new_column = column.map(map_val_to_ind)
        return new_column

    def calc_save_global_index_value_mapping(self, dataframe):
        global_index_value_mapping = {}
        global_value_index_mapping = {}
        offset = 0
        for column_name in dataframe.columns:
            if column_name != label:
                value_index_mapping = self.calc_index_value_mapping_from_column(dataframe[column_name], offset)
                offset += len(value_index_mapping)
                curr_column_items = value_index_mapping.items()
                global_index_value_mapping[column_name] = {index: value for value, index in curr_column_items}
                global_value_index_mapping[column_name] = {value: index for value, index in curr_column_items}

        self.save_index_value_mapping(global_index_value_mapping)
        self.save_value_index_mapping(global_value_index_mapping)

        return global_index_value_mapping, global_value_index_mapping

    def index_df(self, dataframe):
        new_dataframe = pd.DataFrame()
        # global_index_value_mapping, global_value_index_mapping = calc_global_index_value_mapping(dataframe)
        global_value_index_mapping = self.load_value_index_mapping()
        for column_name in dataframe.columns:
            if column_name != label:
                value_index_mapping = global_value_index_mapping[column_name]
                new_dataframe[column_name] = self.index_column_from_mapping(dataframe[column_name], value_index_mapping)
            else:
                new_dataframe[column_name] = dataframe[column_name]

        return new_dataframe

    def fit(self, train_dataset_path):
        df = self.read_dataset(train_dataset_path)

        # df.describe(include=(np.number))
        # df.describe(include=[object])

        self.calc_save_frequent_values(df)
        frequent_values = self.load_frequent_values()
        self.replace_infrequent_with_default(df, frequent_values)
        self.replace_numeric_with_bins(df)
        self.calc_save_global_index_value_mapping(df)

    def transform(self, dataset_path, save_to_path):
        df = self.read_dataset(dataset_path)

        # Load the frequent values from the file
        frequent_values = self.load_frequent_values()
        # Apply the function to the DataFrame
        self.replace_infrequent_with_default(df, frequent_values)
        self.replace_numeric_with_bins(df)

        new_df = self.index_df(df)  # change

        self.save_dataset(new_df, save_to_path)
