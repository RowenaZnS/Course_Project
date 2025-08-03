#!/bin/bash


#SBATCH --mem=4G
#SBATCH -t 0:30:00

source /data/users/liuxinya/lab3/bin/activate


pip install mrjob
pip install setuptools



python3 assignment3_problem1.py -r local --num-cores 64 /data/courses/2025_dat470_dit066/sc2/planets.csv




