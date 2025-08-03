#!/bin/bash


#SBATCH --cpus-per-task=32


source /data/users/liuxinya/lab3/bin/activate

pip install mrjob
pip install setuptools

python3 mrjob_twitter_follows.py /data/courses/2025_dat470_dit066/twitter/twitter-2010_full.txt




