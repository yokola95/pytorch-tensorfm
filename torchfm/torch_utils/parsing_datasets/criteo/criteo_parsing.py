import pickle
import json
import numpy as np
import pandas as pd
import torch
from math import floor, log

print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available. Version:", torch.version.cuda)
else:
    print("CUDA is not available.")


data_file = "../persistent_drive/data/train20K.txt"  # Replace with the actual path to your dataset file


class CriteoParsing:
    # Set the threshold (minimum frequency) for the default value
    threshold = 10

    # Default value to replace infrequent values
    default_value = 'other_'

    def __init__(self):
        self.cat_names = [f'C{i}' for i in range(1, 27)],
        self.cont_names = [f'I{i}' for i in range(1, 14)],
        self.columns = ['label', *(f'I{i}' for i in range(1, 14)), *(f'C{i}' for i in range(1, 27))]

    def read_dataset(self, data_file_path):
        df = pd.read_csv(data_file_path, sep='\t', names=self.columns)
        return df

    def save_dataset(self, df, save_file_path):
        df.to_csv(save_file_path, header=True, index=False)

    # Function to load frequent values from a file
    def load_frequent_values(self, file_path):
        with open(file_path, 'rb') as f:
            frequent_values = pickle.load(f)
        return frequent_values

    def save_frequent_values(self, file_path, frequent_values):
        # save the frequent values to a file
        with open(file_path, 'wb') as f:
            pickle.dump(frequent_values, f)

    # Function to store frequent values to a file
    def store_frequent_values(self, df, file_path):
        frequent_values = {}
        for col in df.columns:
            if col.startswith('C'):
                # Calculate the frequency of each unique value in the column
                value_counts = df[col].value_counts()
                # Create a set of values that appear at least the threshold times
                frequent_values[col] = set(value_counts[value_counts >= self.threshold].index)

        self.save_frequent_values(self, file_path, frequent_values)

    # Function to replace infrequent values in each column with a default value
    def replace_infrequent_with_default(self, df, frequent_values):
        for col in df.columns:
            if col.startswith('C'):
                # Replace values with the default value based on the frequent values
                mask = (df[col].notnull()) & (~df[col].isin(frequent_values[col]))
                df.loc[mask, col] = self.default_value + col
                df[col].fillna('SP_NaN', inplace=True)

    # Save the index-value mapping to a file (debug)
    def save_index_value_mapping(self, global_index_value_mapping):
        with open(data_file + '.index2value.json', 'w') as mapping_file:
            json.dump(global_index_value_mapping, mapping_file)

    # Save the value-index mapping to a file (debug)
    def save_value_index_mapping(self, global_value_index_mapping):
        with open(data_file + '.value2index.json', 'w') as mapping_file:
            json.dump(global_value_index_mapping, mapping_file)

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

        if 'label' in numerical_cols:
            numerical_cols = numerical_cols.drop('label')

        # Apply the custom transformation to the numerical columns
        df[numerical_cols] = df[numerical_cols].applymap(numeric_transform)

    def fit(self, data_file):
        df = self.read_dataset(data_file)

        # df.describe(include=(np.number))
        # df.describe(include=[object])

        # Apply the function to store frequent values to a file
        self.store_frequent_values(df, data_file + '.frequent_values.pkl')

        # Load the frequent values from the file
        frequent_values = self.load_frequent_values(data_file + '.frequent_values.pkl')

        # Apply the function to the DataFrame
        self.replace_infrequent_with_default(df, frequent_values)

        return df

    def index_column(self, column, offset):
        unique_values = column.unique()
        value_index_mapping = {}
        value_index_mapping['unknown'] = offset
        value_index_mapping.update({value: offset + index + 1 for index, value in enumerate(unique_values)})

        new_column = column.map(value_index_mapping)
        return new_column, value_index_mapping

    def index_df(self, df):
        new_dataframe = pd.DataFrame()
        offset = 0
        global_index_value_mapping = {}
        global_value_index_mapping = {}
        for column_name in df.columns:
            if column_name != 'label':
                new_column, value_index_mapping = self.index_column(df[column_name], offset)
                new_dataframe[column_name] = new_column
                offset += len(value_index_mapping)
                global_index_value_mapping.update(
                    {column_name + '>' + str(index): value for value, index in value_index_mapping.items()})
                global_value_index_mapping.update(
                    {column_name + '>' + value: index for value, index in value_index_mapping.items()})
            else:
                new_dataframe[column_name] = df[column_name]

        # Save the index-value mapping to a file (debug)
        self.save_index_value_mapping(global_index_value_mapping)

        # Save the value-index mapping to a file (debug)
        self.save_value_index_mapping(global_value_index_mapping)
        return new_dataframe

    def transform(self, dataset_path):
        df = self.read_dataset(dataset_path)
        # Apply the function to the DataFrame
        self.replace_numeric_with_bins(df)