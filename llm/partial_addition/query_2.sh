#!/bin/bash -l
#SBATCH -o /your/path/llm/partial_addition/logs/partial_%A_%a.out
#SBATCH -e /your/path/llm/partial_addition/logs/partial_%A_%a.err
#SBATCH --job-name=partial_addition_2
#SBATCH --time=100:00:00
#SBATCH --array=0
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --cpus-per-task=18
#SBATCH --constraint="gpu"

# --- start from a clean state and load necessary environment modules ---
module purge
##module load cuda/11.6
module load anaconda/3/2023.03
conda activate ml

## This training is used in order to retrain the models that somehow didn't finish training due to memory error.

# Set the number of cores available per process if the $SLURM_CPUS_PER_TASK is set
if [ ! -z $SLURM_CPUS_PER_TASK ] ; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=1
fi

## cd '/your/path/llm/partial_addition/'

python query.py --success 0.5 --fail -0.5 --num_options 3

python query.py --success 1.0 --fail 0.0 --num_options 4
python query.py --success 0.5 --fail -0.5 --num_options 4
