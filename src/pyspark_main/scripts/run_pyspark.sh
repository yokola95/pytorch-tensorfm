# Explanations about the docker image: https://jetblue-jupyter.blue.ygrid.yahoo.com:9999/nb/notebooks/projects/notebooks/jupyter/samples[…]/Jupyter_Demo_2.0__Custom_Python_Libraries.ipynb

# ml image docker doc: https://git.ouryahoo.com/hadoop-user-images/ml/blob/master/image-deploy.yaml
# https://tools.ygrid.yahoo.com:4443/abft/#/status/huserimages

# Ariel's project  https://git.ouryahoo.com/haifa-labs-mail/Geonosis/tree/dev

export QUEUE=default
export BASE_PATH=hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/viderman/low_rank_experiments
export SRC_PATH=${BASE_PATH}/src
# ------------------------------------- #

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--executor-memory 10G \
--driver-memory 15G \
--conf spark.oath.dockerImage=ml/rhel8_mlbundle:2021.12.1 \
--conf spark.driver.memoryOverhead=10G \
--conf spark.executor.memoryOverhead=6G \
--conf spark.hadoop.hive.metastore.uris=thrift://bassniumblue-hcat.blue.ygrid.yahoo.com:50513 \
--conf spark.kerberos.access.hadoopFileSystems=hdfs://bassniumblue-nn1.blue.ygrid.yahoo.com:8020 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.ui.view.acls=* \
--archives ${SRC_PATH}/torchmetrics.zip \
--py-files ${SRC_PATH}/src.zip \
${SRC_PATH}/run_pyspark_experiment.py

#--conf spark.driver.resource.gpu.amount=1 \
#
