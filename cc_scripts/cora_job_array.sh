#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=00:15:00     # time of the task
#SBATCH --account=def-lelis
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=emireddy@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-816

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.9
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install torch --no-index
pip install torch_sparse --no-index
pip install torch_scatter --no-index
pip install torch_geometric --no-index
pip install node2vec --no-index
pip install -r requirements.txt 

module load cuda cudnn


python3 ../experiments.py -n cora exec --id $SLURM_ARRAY_TASK_ID


