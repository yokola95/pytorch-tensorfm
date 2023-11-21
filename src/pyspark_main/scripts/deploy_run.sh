hkinitall
chmod 775 src.zip run_pyspark_experiment.py
ascfs hput -f src.zip run_pyspark_experiment.py hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/viderman/low_rank_experiments/src/
ascfs hchmod 775 hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/viderman/low_rank_experiments/src/*

rm src.zip run_pyspark_experiment.py

ascfs ./run_pyspark.sh
