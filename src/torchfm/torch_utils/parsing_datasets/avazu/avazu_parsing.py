import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from src.torchfm.torch_utils.constants import test_datasets_path


class AvazuParsing:
    avazu_path = f"{test_datasets_path}/data_avazu.csv"
    threshold = 10
    testRatio = 0.1
    valRatio = 0.1
    train_raw_path = f"{test_datasets_path}/train_raw.csv"
    test_raw_path = f"{test_datasets_path}/test_raw.csv"
    val_raw_path = f"{test_datasets_path}/val_raw.csv"

    def swapColumns(self, dataFrame, col1, col2):
        cols = dataFrame.columns.tolist()
        colId_idx = cols.index(col1)
        colClick_idx = cols.index(col2)
        cols[colId_idx], cols[colClick_idx] = cols[colClick_idx], cols[colId_idx]
        return dataFrame[cols]

    def getHour(self, time):
        return int(time % 100)

    def getDay(self, time):
        return int((time / 100) % 100)

    # Function to replace infrequent values in each column with a default value
    def replace_infrequent_with_default(self, df, frequent_values):
        for col in df.columns:
            if col != 'click' and col != 'id' and col != 'label':
                # Replace values with the default value based on the frequent values
                mask = (df[col].notnull()) & (~df[col].isin(frequent_values[col]))
                df.loc[mask, col] = "rare"

                df[col].fillna('SP_NaN', inplace=True)

    # Function to store frequent values to a file
    def store_frequent_values(self, df, threshold, file_path):
        frequent_values = {}
        for col in df.columns:
            if col != 'click' and col != 'id' and col != 'label':
                # Calculate the frequency of each unique value in the column
                value_counts = df[col].value_counts()

                # Create a set of values that appear at least the threshold times
                frequent_values[col] = set(value_counts[value_counts >= threshold].index)

            # Store the frequent values to a file
            with open(file_path, 'wb') as f:
                pickle.dump(frequent_values, f)

    # Function to load frequent values from a file
    def load_frequent_values(self, file_path):
        with open(file_path, 'rb') as f:
            frequent_values = pickle.load(f)
        return frequent_values

    def read_frequent(self, df, file_name):
        # Load the frequent values from the file
        frequent_values = self.load_frequent_values(file_name)

        # Apply the function to the DataFrame
        self.replace_infrequent_with_default(df, frequent_values)

    def keep_frequent(self, df, threshold):
        # Apply the function to store frequent values to a file
        self.store_frequent_values(df, threshold, f"{test_datasets_path}/frequent_values.pkl")

        self.read_frequent(df, f"{test_datasets_path}/frequent_values.pkl")

    def save_dic(self, dic, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    def read_dic(self, file_name):
        with open(file_name, 'rb') as file:
            loaded_dict = pickle.load(file)

            return loaded_dict

    def save_index_column(self, column, offset):
        unique_values = column.unique()
        value_index_mapping = {}
        value_index_mapping['unknown'] = offset
        value_index_mapping["rare"] = offset + 1
        value_index_mapping.update({value: offset + index + 2 for index, value in enumerate(unique_values)})

        return value_index_mapping

    def save_global_index(self, dataframe):
        new_dataframe = pd.DataFrame()
        offset = 0
        global_index_value_mapping = {}
        global_value_index_mapping = {}
        for column_name in dataframe.columns:
            if column_name != 'click' and column_name != 'id' and column_name != 'label':
                value_index_mapping = self.save_index_column(dataframe[column_name], offset)
                offset += len(value_index_mapping)
                global_index_value_mapping[column_name] = {index: value for value, index in value_index_mapping.items()}
                global_value_index_mapping[column_name] = {value: index for value, index in value_index_mapping.items()}

        self.save_dic(global_index_value_mapping, f"{test_datasets_path}/global_index_value_mapping")
        self.save_dic(global_value_index_mapping, f"{test_datasets_path}/global_value_index_mapping")

    def preprocess_before_split(self, db):
        db['label'] = db['click'].astype(int)
        db.drop(columns=['click', 'id'], inplace=True)
        #        db["day"] = db["hour"].apply(lambda x: self.getDay(x))
        #        db["hour"] = db["hour"].apply(lambda x: self.getHour(x))

        return db

    def index_column_from_mapping(self, column, value_index_mapping):
        def map_val_to_ind(x):
            if x in value_index_mapping:
                return value_index_mapping[x]
            elif (x is None) or (x == ''):  # (is_numeric_dtype(column) and np.isnan(x)) or
                return value_index_mapping["missing"]
            else:
                return value_index_mapping["rare"]

        new_column = column.map(map_val_to_ind)  # new_column = column.map(value_index_mapping)
        return new_column

    def index_df(self, dataframe):
        new_dataframe = pd.DataFrame()
        global_value_index_mapping = self.read_dic(f"{test_datasets_path}//global_value_index_mapping")
        for column_name in dataframe.columns:
            if column_name != 'click' and column_name != 'id' and column_name != 'label':
                value_index_mapping = global_value_index_mapping[column_name]
                new_dataframe[column_name] = self.index_column_from_mapping(dataframe[column_name], value_index_mapping)
            else:
                new_dataframe[column_name] = dataframe[column_name]

        return new_dataframe

    def splitAndWrite(self, data, train_data_path, test_data_path, val_data_path):
        self.preprocess_before_split(data)
        testRatio = self.testRatio
        valRatio = self.valRatio

        train_data, temp_data = train_test_split(data, test_size=(testRatio + valRatio), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=(testRatio / (testRatio + valRatio)),
                                               random_state=42)
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        val_data.to_csv(val_data_path, index=False)

    def split(self, db):
        self.splitAndWrite(db, self.train_raw_path, self.test_raw_path, self.val_raw_path)

    def fit(self, db):
        self.keep_frequent(db, self.threshold)
        self.save_global_index(db)

    def transform(self, db, final_path):
        self.read_frequent(db, f"{test_datasets_path}/frequent_values.pkl")
        db = self.index_df(db)
        db.to_csv(final_path, index=False)

    def read_dataset_orig(self, data_file_path):  # no header
        df = pd.read_csv(data_file_path)
        return df

    # def read_dataset(self, data_file_path):  # with header
    #     df = pd.read_csv(data_file_path, sep='\t', header='infer')
    #     return df


def process_data():
    avazu_parsing = AvazuParsing()
    raw_data = avazu_parsing.read_dataset_orig(avazu_parsing.avazu_path)
    avazu_parsing.split(raw_data)

    raw_train = avazu_parsing.read_dataset_orig(avazu_parsing.train_raw_path)
    raw_val = avazu_parsing.read_dataset_orig(avazu_parsing.val_raw_path)
    raw_test = avazu_parsing.read_dataset_orig(avazu_parsing.test_raw_path)

    avazu_parsing.fit(raw_train)

    avazu_parsing.transform(raw_train, f"{test_datasets_path}/train.csv")
    avazu_parsing.transform(raw_test, f"{test_datasets_path}/test.csv")
    avazu_parsing.transform(raw_val, f"{test_datasets_path}/validation.csv")


# process_data()
