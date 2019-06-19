#!/bin/bash

#SBATCH --job-name=sim-tr2
#SBATCH --output=log_simulate_train2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

#python -u simulate.py --fixz --fixm --fixalign -n 10000 --name train_fix_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
#python -u simulate.py --fixz --fixalign -n 10000 --name train_mass_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --fixz --fixm -n 10000 --name train_align_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
#python -u simulate.py -n 10000 --name train_full_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
