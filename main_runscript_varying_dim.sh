#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p sapphire,unrestricted,janson,janson_cascade,shared,serial_requeue
#SBATCH -o outputs/output_%A_%a.out
#SBATCH -e errors/errors_%A_%a.err
#SBATCH -t 0-10:30 
#SBATCH --array=1-20

module load python/3.10.13-fasrc01
mamba run -n yash python3 testing_driver.py 50 100 ${SLURM_ARRAY_TASK_ID} $1 False $2 $3 $4 100 6 $5
