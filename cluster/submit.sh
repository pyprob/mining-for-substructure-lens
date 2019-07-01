#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster


############################################################
# Simulation
############################################################

# sbatch --array=0-99 simulate_train.sh
# sbatch --array=0-624 simulate_calibration.sh
# sbatch simulate_calibration_ref.sh
# sbatch --array=0-9 simulate_test.sh


############################################################
# Combination
############################################################

# sbatch combine_samples.sh


############################################################
# Training
############################################################

# sbatch train_carl.sh
# sbatch train_alices.sh
# sbatch train_carl_full.sh
# sbatch train_alices_full.sh
# sbatch train_carl_exp.sh
# sbatch train_alices_exp.sh
# sbatch train_carl_exp2.sh
# sbatch train_alices_exp2.sh
sbatch train_carl_exp3.sh
sbatch train_alices_exp3.sh
sbatch train_carl_pre.sh
sbatch train_alices_pre.sh


############################################################
# Evaluation
############################################################

# sbatch eval_carl.sh
# sbatch eval_alices.sh
# sbatch eval_carl_full.sh
# sbatch eval_alices_full.sh
# sbatch eval_carl_exp.sh
# sbatch eval_alices_exp.sh
# sbatch eval_carl_full.sh
# sbatch eval_alices_full.sh


############################################################
# Calibration
############################################################

# sbatch calibrate.sh
