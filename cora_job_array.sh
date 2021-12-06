#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M    # memory per node
#SBATCH --time=00:25:00   # time of the task
#SBATCH --account=def-lelis
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=emireddy@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-125
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.9
module load scipy-stack
source ./env/bin/activate
module load cuda cudnn
python3 experiments.py -n lastfm exec --id $SLURM_ARRAY_TASK_ID