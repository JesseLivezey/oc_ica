#!/bin/bash
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Memory:
#SBATCH --mem-per-cpu=15G
#
# Constraint:
#SBATCH --constraint=cortex_k40
#SBATCH --gres=gpu:1
module load cuda
module unload intel

python Development/oc_ica/compare_models.py
