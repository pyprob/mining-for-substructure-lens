#!/bin/bash

#SBATCH --job-name=sim-calpointref
#SBATCH --output=log_simulate_calibration_pointref.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py -n 10000 --calref --pointref --dir /scratch/jb6504/StrongLensing-Inference
