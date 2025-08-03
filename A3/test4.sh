#!/bin/bash


#SBATCH --mem=4G
#SBATCH -t 0:30:00

source /data/users/liuxinya/lab3/bin/activate

pip install mrjob
pip install setuptools

#python3 mrjob_twitter_followers.py /data/courses/2025_dat470_dit066/twitter/twitter-2010_1k.txt
#python3 mrjob_twitter_followers.py /data/courses/2025_dat470_dit066/twitter/twitter-2010_10k.txt
#python3 mrjob_twitter_followers.py /data/courses/2025_dat470_dit066/twitter/twitter-2010_100k.txt
python3 mrjob_twitter_followers.py /data/courses/2025_dat470_dit066/twitter/twitter-2010_full.txt




