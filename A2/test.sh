#! /bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0:30:00


source /data/users/leahw/lab2/.venv/bin/activate

#echo output from assignment2_problem2_skeleton.py
#echo this is tiny
#python3 assignment2_problem2a.py /data/courses/2025_dat470_dit066/gutenberg/tiny
#echo this is small
#python3 assignment2_problem2a.py /data/courses/2025_dat470_dit066/gutenberg/small
#echo this is medium
#python3 assignment2_problem2a.py /data/courses/2025_dat470_dit066/gutenberg/medium
#echo this is big
#python3 assignment2_problem2a.py /data/courses/2025_dat470_dit066/gutenberg/big
#echo this is huge
#python3 assignment2_problem2a.py /data/courses/2025_dat470_dit066/gutenberg/huge

echo task 3
python3 assignment2_problem2d.py -w 32 -b 1 /data/courses/2025_dat470_dit066/gutenberg/huge
#pip install matplotlib
#python3 assignment2_problem2e.py /data/courses/2025_dat470_dit066/gutenberg/tiny
