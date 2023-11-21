zip -r src.zip ./src
rsync src.zip src/pyspark_main/scripts/run_pyspark_experiment.py jet-gw.blue.ygrid.yahoo.com:~/
rm src.zip
