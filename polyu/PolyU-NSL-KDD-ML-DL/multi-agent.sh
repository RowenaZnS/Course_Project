#! /bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L4:1

source /data/users/liuxinya/miniforge3/bin/activate pyspark

python3 multi_agent.py
