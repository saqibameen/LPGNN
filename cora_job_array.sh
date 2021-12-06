#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=01:00:00     # time of the task
#SBATCH --account=rrg-lelis
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=saqib1@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-816

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.9
module load scipy-stack

source ./env/bin/activate
# module load cuda cudnn


python3 experiments.py -n lastfm exec --id $SLURM_ARRAY_TASK_ID