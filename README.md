# Factorization Machine models in PyTorch
# The code was forked and modified from https://github.com/rixwew/pytorch-fm 
# The original code was written by rixwew. The API Documentation of the original code is: https://rixwew.github.io/pytorch-fm

This package provides a PyTorch implementation of low rank factorization machine models, the factorization model baselines and the common datasets in CTR prediction.


## Available Datasets

* [MovieLens Dataset](https://grouplens.org/datasets/movielens)
* [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
* [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)

## Data Processing

Summary for data processing:
All features are categorical
Rare elements (need to define) are replaced by an additional ‘dummy’ value. Rare elements are defined as those that appear less than N times (a configurable threshold)
Numerical features are categorized (binning) using [0, 1, floor(log^2(x))]. If the numerical range does not fit this strategy, let’s talk.
Features are numbered globally from 0 to n-1 (n is the number of unique values, including an ‘unknown’ value for each column)
All columns have an allocated missing value index, even if the column does not contain missing values. This is for the case that we observed a new value in the test set and we need to map it somewhere.

Separator between the fields: ,
Secondary separator for the multilabel features: |

Example of preprocessed dataset:

label, user_id, item_id, C1, C2, …
<label>,10,2,3,17,11,15
<label>,11,2,4,16,12,14


# How to preprocess Avazu dataset


1. Download the initial file (train, 6.31G) should be called data_avazu.csv
   https://www.kaggle.com/datasets/atirpetkar/avazu-ctr

2. put it to pytorch-fm/torchfm/test-datasets/avazu/

3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.avazu.avazu_parsing import process_data 
    process_data()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-fm/torchfm/test-datasets/avazu/

5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.

# How to preprocess Criteo dataset

1. Download the initial file (train.txt, 11.15G) should be called data_criteo.csv
   https://www.kaggle.com/datasets/mrkmakr/criteo-dataset
2. put it to pytorch-fm/torchfm/test-datasets/criteo/
3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.criteo.criteo_parsing import CriteoParsing 
    CriteoParsing.do_preprocessing()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-fm/torchfm/test-datasets/criteo/
5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.

# How to preprocess MovieLens dataset
1. Download the initial file users.dat, movies.dat, ratings.dat from
   https://www.kaggle.com/datasets/sherinclaudia/movielens
2. put it to pytorch-fm/torchfm/test-datasets/movielens/
3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.movielens.movielens_parsing import MovielensParsing 
    MovielensParsing.process_data()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-fm/torchfm/test-datasets/movielens/
5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.


# Instructions: How to run a dataset on ephemeral instance

1. Copy the code under the persistent drive folder. Install all requirements. 

2. In the shell redefine PYTHONPATH to point to your project root, .e.g,
export PYTHONPATH=$PYTHONPATH:/home/${USER}/persistent_drive/pytorch-fm/src:/home/${USER}/persistent_drive/pytorch-fm/src/main_functions

3. Edit pytorch-fm/torchfm/torch_utils/constants.py file 
to have 
      - base_path_project pointing to your project root
      - Edit dataset_name to have the dataset you run on (avazu, criteo, movielens)
        E.g., dataset_name = movielens

Edit pytorch-fm/src/main_functions/run_processes.py file in examples folder, 
to refer to the list of options to run:  currently, as example, it contains lst_tmp
in "for tpl in lst_tmp:"
 
4. Copy train.csv/validation.csv/test.csv splitted datasets to be under pytorch-fm/data/test-datasets/<dataset>/
(dataset is either criteo or avazu or movielens)
Then, open a python shell by running just: python command from the ephemeral shell.

5. Check that you have enough space (at least 5G available) after all these steps in ephemeral by:
df -h /home/default/persistent_drive
Otherwise remove non-required data (e.g., datasources you don’t use for the current run)

6. Check that you have pytorch-fm/data/tmp_save_dir (if not create this folder: mkdir)

7. If you are rerunning, check you don’t have a previous run. results stored (especially .log files - locking the next run), otherwise remove:
rm pytorch-fm/data/tmp_save_dir/*

8. Run by 
python ./pytorch-fm/src/main_functions/run_processes.py

9. After the run is done, the results are saved in 
persistent_drive/pytorch-fm/data/tmp_save_dir/optuna_results.txt 
Also, debug info is saved in persistent_drive/pytorch-fm/data/tmp_save_dir/debug_info.txt

   
