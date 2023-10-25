import argparse
import logging
import random
import os
import pickle
from datetime import datetime
from math import ceil
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import row_number, lit, col
from pyspark.sql.window import Window

from main_functions.main import top_main_for_option_run
from options_to_run import all_options_for_studies, Option2Run

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, required=False, default=15)
parser.add_argument("--batch_size", type=int, required=False, default=32)
parser.add_argument("--hidden_size", type=int, required=False, default=512)

df_columns = ["model", "metric_to_opt", "rank_prm", "emb_size", "lr", "opt_name", "batch_size", "return_l2", "reg_coef_vectors", "reg_coef_biases"]


def create_df_options_to_run(spark : SparkSession) -> DataFrame:
    df = spark.createDataFrame(data=all_options_for_studies, schema=df_columns)
    return df.withColumn("row_id", row_number().over(Window.orderBy(df_columns))).repartition("row_id")


def run_training_models(spark: SparkSession):
    def process_data(partition_data):
        r = random.random()
        for row in partition_data:
            option2run = Option2Run(row.model, row.metric_to_opt, row.rank_prm, row.emb_size, row.lr, row.opt_name, row.batch_size, row.return_l2, row.reg_coef_vectors, row.reg_coef_biases)
            top_main_for_option_run(None, None, 0, option2run)
            yield [row.model, row.metric_to_opt, row.rank_prm, row.emb_size, row.lr, row.opt_name, row.batch_size, row.return_l2, row.reg_coef_vectors, row.reg_coef_biases, row.row_id, r]

    df = create_df_options_to_run(spark)
    df.rdd.mapPartitions(process_data).toDF(df_columns + ["row_id", "random"]).sort("row_id").show(n=100)


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .appName("FM models Training") \
        .enableHiveSupport() \
        .getOrCreate()

    # print("CUDA_VISIBLE_DEVICES={}".format(os.environ['CUDA_VISIBLE_DEVICES']))
    # print(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))))

    # parse arguments
    args: argparse.Namespace = parser.parse_args()

    # log arguments
    arguments_logline = "Arguments passed to main:\n"
    arguments_logline += str(args)
    arguments_logline += "arguments --------------------------------------- \n"
    logging.info(arguments_logline)
    print(arguments_logline)

    # run models
    run_training_models(spark)
