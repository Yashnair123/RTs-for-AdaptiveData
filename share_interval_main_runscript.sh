#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p # Partition to submit to
#SBATCH -o outputs/output_%A_%a.out
#SBATCH -e errors/errors_%A_%a.err
#SBATCH -t 0-05:00 
#SBATCH --array=1-50


module load python/3.10.13-fasrc01
mamba run -n <conda-environment> python3 conformal_interval_driver_share.py 20 $1 ${SLURM_ARRAY_TASK_ID} $2 $3 $4 $5
