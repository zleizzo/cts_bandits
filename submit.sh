#!/bin/bash

#SBATCH --job-name=cohort
#SBATCH --output=/home/users/zizzo/log/cohort_%A_%a.out
#SBATCH --error=/home/users/zizzo/log/cohort_%A_%a.err
#SBATCH --array=0-720
#SBATCH --time=5:59:00
#SBATCH -p normal,jamesz
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=20G


######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

# Run your code based on the SLURM_ARRAY_TASK_ID
# For example: 
# ml python/3.6.1
# conda activate cohort
# python3 run_experiment.py $SLURM_ARRAY_TASK_ID
python3 experiment_with_t.py $SLURM_ARRAY_TASK_ID