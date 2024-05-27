#!/bin/bash -l
#SBATCH -o ./logs/fits_%A_%a.out
#SBATCH -e ./logs/fits_%A_%a.err
#SBATCH --job-name=fits
#SBATCH --time=100:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=18
#SBATCH --array=0-7

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

## cd "/cognitive_models/"


declare -a part_list

part_list+=("partial_main"
            "full"
            "agency"
            "partial_llms"
            "partial_addition")


parameters=(${part_list[${SLURM_ARRAY_TASK_ID}]})
fit=${parameters[0]}

echo "fit: ${fit}"

python3 fitting.py --fits $fit