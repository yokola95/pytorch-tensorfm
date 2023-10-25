# Explanations about the docker image: https://jetblue-jupyter.blue.ygrid.yahoo.com:9999/nb/notebooks/projects/notebooks/jupyter/samples[…]/Jupyter_Demo_2.0__Custom_Python_Libraries.ipynb

# ml image docker doc: https://git.ouryahoo.com/hadoop-user-images/ml/blob/master/image-deploy.yaml
# https://tools.ygrid.yahoo.com:4443/abft/#/status/huserimages

# Ariel's project  https://git.ouryahoo.com/haifa-labs-mail/Geonosis/tree/dev

export QUEUE=gpu_v100
export BASE_PATH=hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/magma/ariell/geonosis_pipeline
export SRC_PATH=${BASE_PATH}/src
export MODEL_NAME="geonosis_v1"
export MODEL_DIR_PATH=${BASE_PATH}/model
export DATASET_PATH=${BASE_PATH}/dataset
export TRAIN_DATA_PATH=${DATASET_PATH}/train_data_path
export DEV_DATA_PATH=${DATASET_PATH}/dev_data_path
export LOGFILE_DIR=${DATASET_PATH}/logfiles

# -------- Training Parameters -------- #

export EPOCHS=$1
export LEARNING_RATE=0.01
export MAX_LENGTH=15
export BATCH_SIZE=32
export EARLY_STOPPING_PATIENCE=10
export EMBEDDING_OUTPUT_SIZE=32
export HIDDEN_SIZE=512
export DROPOUT=0.1
export VERBOSE=true

# ------------------------------------- #


${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--executor-memory 4G \
--driver-memory 10G \
--conf spark.oath.dockerImage=ml/rhel8_mlbundle:2021.12.1 \
--conf spark.driver.resource.gpu.amount=1 \
--conf spark.driver.memoryOverhead=10G \
--conf spark.executor.memoryOverhead=6G \
--conf spark.hadoop.hive.metastore.uris=thrift://bassniumblue-hcat.blue.ygrid.yahoo.com:50513 \
--conf spark.kerberos.access.hadoopFileSystems=hdfs://bassniumblue-nn1.blue.ygrid.yahoo.com:8020 \
--conf spark.yarn.access.namenodes=hdfs:/bassniumblue-nn1.blue.ygrid.yahoo.com:8020 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.ui.view.acls=* \
--archives ${SRC_PATH}/bs4.zip \
--py-files ${SRC_PATH}/geonosis_src.zip \
${SRC_PATH}/model_training.py \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --embedding_output_size ${EMBEDDING_OUTPUT_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --dropout ${DROPOUT} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --verbose ${VERBOSE} \
		--model_name ${MODEL_NAME} \
		--model_dir_base_path ${MODEL_DIR_PATH} \
		--train_data_path ${TRAIN_DATA_PATH} \
		--dev_data_path ${DEV_DATA_PATH} \
		--epochs ${EPOCHS} \
		--logfile_dir ${LOGFILE_DIR}
