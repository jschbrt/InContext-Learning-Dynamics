#!/bin/bash -l
#SBATCH -o ./logs/train_masking_%A_%a.out
#SBATCH -e ./logs/train_masking_%A_%a.err
#SBATCH --job-name=transformer
#SBATCH --time=100:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=18
#SBATCH --array=0

# --- start from a clean state and load necessary environment modules ---
module purge
##module load cuda/11.6
module load anaconda/3/2023.03
conda activate ml


## This training is used in order to retrain the models that somehow didn't finish training due to memory error.
## List was generated with exp012 inspect_failed_500k_rc_efv_jobs.ipynb

# Set the number of cores available per process if the $SLURM_CPUS_PER_TASK is set
if [ ! -z $SLURM_CPUS_PER_TASK ] ; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=1
fi

##cd 'path to meta-rl/agency/task/'


declare -a part_list

part_list+=("10000 64 mask_policy_value_loss")

parameters=(${part_list[${SLURM_ARRAY_TASK_ID}]})
train_eps=${parameters[0]}
batch_size=${parameters[1]}
agency_test=${parameters[2]}

echo "train_eps: ${train_eps}"
echo "batch_size: ${batch_size}"
echo "agency_test: ${agency_test}"

model_name="gpu_eps${train_eps}_bs${batch_size}_agency_test_${agency_test}"
exp_name="exp"

python3 train.py --exp_name $exp_name --model_name $model_name --train_eps $train_eps --batch_size $batch_size --agency_test $agency_test --task-id $SLURM_ARRAY_TASK_ID --subject $SLURM_ARRAY_TASK_ID
