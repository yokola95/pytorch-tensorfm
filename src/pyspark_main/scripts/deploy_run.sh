zip -r src.zip ./src
rsync src.zip src/pyspark_main/scripts/run_pyspark.sh src/pyspark_main/scripts/run_pyspark_experiment.py jet-gw.blue.ygrid.yahoo.com:~/
rm src.zip

ssh -A ${USER}@jet-gw.blue.ygrid.yahoo.com "pkinit-user; sudo -u cfs_head pkinit-user;
chmod 775 src.zip run_pyspark.sh run_pyspark_experiment.py
ascfs hput -f src.zip run_pyspark_experiment.py hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/${USER}/low_rank_experiments/src/
ascfs hchmod 775 hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/${USER}/low_rank_experiments/src/*
rm src.zip run_pyspark_experiment.py
./run_pyspark.sh"

