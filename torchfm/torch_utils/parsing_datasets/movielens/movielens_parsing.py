import pandas as pd
import os

def load_and_merge_ml(ml_dir):
    # load MovieLens data using encoding="ISO-8859-1" to avoid UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9

    # users
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(ml_dir + '/users.dat', sep='::',
                          header=None, names=unames, engine='python', encoding="ISO-8859-1")
    print('number of unique users:', users['user_id'].unique().size)

    # ratings
    rnames = ['user_id', 'movie_id', 'label', 'timestamp']
    ratings = pd.read_table(ml_dir + '/ratings.dat', sep='::',header=None, names=rnames, engine='python', encoding="ISO-8859-1")
    human_time = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['year'] = human_time.dt.year
    ratings['month'] = human_time.dt.month
    ratings['day_of_week'] = human_time.dt.dayofweek
    ratings['hour'] = human_time.dt.hour
    ratings.drop('timestamp', axis='columns', inplace=True)
    print('number of ratings:', len(ratings))

    # movies
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(ml_dir + '/movies.dat', sep='::',header=None, names=mnames, engine='python', encoding="ISO-8859-1")
    print('number of unique movies:', movies['movie_id'].unique().size)

    # merge data
    merged_ml = ratings.merge(users, on='user_id')
    merged_ml = merged_ml.merge(movies, on='movie_id')
    print('number of NaN values in the merged table ', merged_ml.isna().sum().sum())

    # move label to be the first column
    colmuns = merged_ml.columns.tolist()
    colmuns.insert(0, colmuns.pop(colmuns.index('label')))
    merged_ml = merged_ml[colmuns]

    # drop title column
    merged_ml = merged_ml.drop('title', axis='columns')

    return merged_ml


def split_ml(merged_ml, split_ratios):
    train_ml, validation_ml, test_ml = [], [], []
    if (abs(sum(split_ratios.values()) - 1) < 1e-5) and all(v > 0.05 for v in split_ratios.values()):
        print('split values are ok')
        train_ml = merged_ml.sample(frac=split_ratios['train'], axis=0)
        validation_ml = merged_ml.drop(train_ml.index).sample(
            frac=split_ratios['validation'] / (1 - split_ratios['train']), axis=0)
        test_ml = merged_ml.drop(train_ml.index).drop(validation_ml.index)
        if ((train_ml.shape[0] + validation_ml.shape[0] + test_ml.shape[0]) == merged_ml.shape[0]) and (
                train_ml.shape[1] == merged_ml.shape[1]) and (validation_ml.shape[1] == merged_ml.shape[1]) and (
                test_ml.shape[1] == merged_ml.shape[1]):
            print('split is ok')
            print('merged:', merged_ml.shape, 'train:', train_ml.shape, 'validation:', validation_ml.shape, 'test:',
                  test_ml.shape)
        else:
            print('split is problematic')
    else:
        print('split values are problematic')
        print(split_ratios)

    return train_ml, validation_ml, test_ml


def analyze_ml(df, min_occurrences):
    ordinal_mapping = []
    feature_index = 0
    total_missing_values_count = 0
    total_unique_values_count = 0
    total_rare_values_count = 0
    print('------------------')

    for column_name in df.columns.tolist():
        if column_name == 'label':
            continue
        print('column:', column_name)
        missing_values_count = df[column_name].isna().sum()
        print('number of missing values:', missing_values_count)
        total_missing_values_count += missing_values_count

        if column_name == 'genres':
            value_list = []
            for movie_genres in df[column_name].values:
                if movie_genres == '': continue
                value_list.extend(movie_genres.split('|'))
            values_freq = dict(pd.Series(value_list).value_counts())  # value_counts ignores NaN
        else:
            values_freq = dict(df[column_name].value_counts())  # value_counts ignores NaN

        rare_values = []
        for value in values_freq.keys():
            if values_freq.get(value) < min_occurrences:
                rare_values.append(value)
        [values_freq.pop(key) for key in rare_values]

        unique_values_count = len(values_freq)
        total_unique_values_count += unique_values_count
        print('number of unique entries:', unique_values_count)
        rare_values_count = len(rare_values)
        total_rare_values_count += rare_values_count
        print('number of rare value:', rare_values_count)

        # generate ordinal encoding for the current column
        column_ordinal_mapping = {k: i for i, k in enumerate(values_freq.keys(), feature_index)}
        # add 'RARE_VALUE' and 'MISSING_VALUE' to the ordinal encoding
        column_ordinal_mapping['RARE_VALUE'] = feature_index + len(column_ordinal_mapping)
        column_ordinal_mapping['MISSING_VALUE'] = feature_index + len(column_ordinal_mapping) + 1
        # add the column ordinal encoding to the list of ordinal encoding
        ordinal_mapping.append(column_ordinal_mapping)
        # update the feature count
        feature_index += len(column_ordinal_mapping)
        print('num of features:', feature_index, 'size of ordinal encoding')
        print('------------------')

    print('total uniqu values count:', total_unique_values_count)
    print('total missing values count:', total_missing_values_count)
    print('total rare values count:', total_rare_values_count)

    return ordinal_mapping


def transform_ml(df, ordinal_encoding):
    encoding_index = 0
    total_missing_values_count = 0
    total_rare_values_count = 0
    line_count = len(df.label)
    print('------------------')
    for column_name in train_ml.columns.tolist():
        if column_name == 'label':
            continue
        print('column:', column_name, encoding_index)
        # replace NaN with 'MISSING_VALUE'
        missing_values_count = df[column_name].isna().sum()
        print('number of missing values:', missing_values_count)
        total_missing_values_count += missing_values_count
        df[column_name].fillna('MISSING_VALUE', inplace=True)
        if column_name == 'genres':
            for i in range(len(df[column_name])):
                genres_list = df['genres'].values[i].split('|')
                new_genres_str = ''
                rare_genre = ordinal_encoding[encoding_index].get('RARE_VALUE')
                is_rare_genre_not_included = True
                for genre in genres_list:
                    temp_genre = ordinal_encoding[encoding_index].get(genre, rare_genre)
                    if is_rare_genre_not_included:
                        new_genres_str += str(temp_genre) + '|'
                    if temp_genre == rare_genre:  # we don't want rare value to be included twice
                        is_rare_genre_not_included = False
                df['genres'].values[i] = new_genres_str[:-1]
        else:
            df[column_name] = df[column_name].map(ordinal_encoding[encoding_index]).fillna(
                ordinal_encoding[encoding_index].get('RARE_VALUE')).astype(int)

        rare_index = ordinal_encoding[encoding_index].get('RARE_VALUE')
        if rare_index in df[column_name].values:
            rare_values_count_column = df[column_name].value_counts()[rare_index]
        else:
            rare_values_count_column = 0
        print('number of rare values:', rare_values_count_column)
        total_rare_values_count += rare_values_count_column
        column_values_count = df[column_name].value_counts().sum()
        print('column values count:', df[column_name].value_counts().sum())
        if line_count != column_values_count:
            print('encoding problem! total value count is', column_values_count, 'and should be', line_count)
            print('program terminated')
            break
        encoding_index += 1
        print('------------------')
    print('total missing values count', total_missing_values_count)
    print('total rare values count', total_rare_values_count)

    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # parameters
    ml1M_dir = '/Users/viderman/Downloads/ml-1m'
    ml1M_trans_dir = '/Users/viderman/Downloads/ml-1m_trans'
    split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    min_occurrences = 10

    # load and merge
    print('load ml and merge files')
    merged_ml = load_and_merge_ml(ml1M_dir)
    # split
    print('split merged ml into train_ml, validation_ml and test_ml')
    train_ml, validation_ml, test_ml = split_ml(merged_ml, split_ratios)
    # analyze
    print('analyze train_ml')
    ordinal_encoding_ml = analyze_ml(train_ml, min_occurrences)
    # transform
    print('transform train_ml')
    train_ml = transform_ml(train_ml, ordinal_encoding_ml)
    print('transform validation_ml')
    validation_ml = transform_ml(validation_ml, ordinal_encoding_ml)
    print('transform test_ml')
    test_ml = transform_ml(test_ml, ordinal_encoding_ml)
    # write csv files
    if not os.path.exists(ml1M_trans_dir):
        os.makedirs(ml1M_trans_dir)
    print('write csv files into',)
    train_ml.to_csv(ml1M_trans_dir + '/train.csv', sep=',', encoding='utf-8', index=False)
    validation_ml.to_csv(ml1M_trans_dir + '/validation.csv', sep=',', encoding='utf-8', index=False)
    test_ml.to_csv(ml1M_trans_dir + '/test.csv', sep=',', encoding='utf-8', index=False)
    print('program completed')
