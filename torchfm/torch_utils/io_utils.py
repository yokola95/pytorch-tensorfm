import torch
import numpy as np
import tarfile
import gzip
import os
import pandas as pd

print(torch.__version__)


if torch.cuda.is_available():
    print("CUDA is available. Version:", torch.version.cuda)
else:
    print("CUDA is not available.")


np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)


def load_dataset(data_file_path):  # criteo
    # Step 2: Load the dataset
    data_file = data_file_path  # "persistent_drive/data/train.txt"  # Replace with the actual path to your dataset file
    cat_names = [f'C{i}' for i in range(1, 27)],
    cont_names = [f'I{i}' for i in range(1, 14)],
    columns = ['label', *(f'I{i}' for i in range(1, 14)), *(f'C{i}' for i in range(1, 27))]
    df = pd.read_csv(data_file, sep='\t', names=columns)
    # df.describe(include=(np.number))
    return df


def data_processing(df):
    import pickle

    # Set the threshold (minimum frequency) for the default value
    threshold = 10

    # Default value to replace infrequent values
    default_value = 'other_'

    # Function to replace infrequent values in each column with a default value
    def replace_infrequent_with_default(df, threshold, default_value, frequent_values):
        for col in df.columns:
            if col.startswith('C'):
                # Replace values with the default value based on the frequent values
                mask = (df[col].notnull()) & (~df[col].isin(frequent_values[col]))
                df.loc[mask, col] = default_value + col

                df[col].fillna('SP_NaN', inplace=True)

    # Function to store frequent values to a file
    def store_frequent_values(df, threshold, file_path):
        frequent_values = {}
        for col in df.columns:
            if col.startswith('C'):
                # Calculate the frequency of each unique value in the column
                value_counts = df[col].value_counts()

                # Create a set of values that appear at least the threshold times
                frequent_values[col] = set(value_counts[value_counts >= threshold].index)

        # Store the frequent values to a file
        with open(file_path, 'wb') as f:
            pickle.dump(frequent_values, f)

    # Function to load frequent values from a file
    def load_frequent_values(file_path):
        with open(file_path, 'rb') as f:
            frequent_values = pickle.load(f)
        return frequent_values

    # Apply the function to store frequent values to a file
    store_frequent_values(df, threshold, data_file + '.frequent_values.pkl')

    # Load the frequent values from the file
    frequent_values = load_frequent_values(data_file + '.frequent_values.pkl')

    # Apply the function to the DataFrame
    replace_infrequent_with_default(df, threshold, default_value, frequent_values)

    from math import floor, log

    def replace_numeric_with_bins(df):
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

    # Apply the function to the DataFrame
    replace_numeric_with_bins(df)
