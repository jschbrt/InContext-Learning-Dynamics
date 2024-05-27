#!/bin/bash -l
#SBATCH -o ../../../logs/tjob_transformer_%A_%a.out
#SBATCH -e ../../../logs/tjob_transformer_%A_%a.err
#SBATCH --job-name=transformer
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

# cd 'path to meta-rl/partial/exp/' 

declare -a task

task=("5000 64")

parameters=(${task[${SLURM_ARRAY_TASK_ID}]})
train_eps=${parameters[0]}
batch_size=${parameters[1]}

echo "train_eps: ${train_eps}"
echo "batch_size: ${batch_size}"
model_name="gpu_eps${train_eps}_bs${batch_size}"
exp_name="exp"

python3 train.py --exp_name $exp_name --model_name $model_name --train_eps $train_eps --batch_size $batch_size --task-id=${SLURM_ARRAY_TASK_ID} --subject ${SLURM_ARRAY_TASK_ID}

## shell script to return_coef this
## sbatch --array=43 experiments/main.sh