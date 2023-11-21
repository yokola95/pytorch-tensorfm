import random

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import spark_partition_id

from src.main_functions.main import top_main_for_option_run
from src.torchfm.torch_utils.options_to_run import all_options_for_studies, Option2Run

df_columns = ["model", "metric_to_opt", "rank_prm", "emb_size", "lr", "opt_name", "batch_size", "reg_coef_vectors", "reg_coef_biases"]


def create_df_options_to_run(spark : SparkSession) -> DataFrame:
    df = spark.createDataFrame(data=all_options_for_studies, schema=df_columns)
    return df.repartition(1000).withColumn("part_id", spark_partition_id())  #.withColumn("row_id", row_number().over(Window.orderBy(df_columns))).repartition("row_id")


def run_training_models(spark: SparkSession):
    def process_data(partition_data):
        for row in partition_data:
            option2run = Option2Run(row.model, row.metric_to_opt, row.rank_prm, row.emb_size, row.lr, row.opt_name, row.batch_size, row.reg_coef_vectors, row.reg_coef_biases, row.part_id)
            top_main_for_option_run(None, None, 0, option2run)
            yield [row.model, row.metric_to_opt, row.rank_prm, row.emb_size, row.lr, row.opt_name, row.batch_size, row.reg_coef_vectors, row.reg_coef_biases]

    df = create_df_options_to_run(spark)
    res_df = df.rdd.mapPartitions(process_data).toDF(df_columns)
    print(res_df.count())


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .appName("FM models Training") \
        .enableHiveSupport() \
        .getOrCreate()

    # print("CUDA_VISIBLE_DEVICES={}".format(os.environ['CUDA_VISIBLE_DEVICES']))
    # print(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))))

    # run models
    run_training_models(spark)
