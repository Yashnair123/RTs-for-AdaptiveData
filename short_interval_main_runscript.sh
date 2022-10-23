#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p unrestricted,shared,serial_requeue
#SBATCH -o outputs/output_%A_%a.out
#SBATCH -e errors/errors_%A_%a.err
#SBATCH -t 0-24:00 
#SBATCH --array=1-20


module purge
module load Anaconda3/2020.11
python interval_driver.py $1 ${SLURM_ARRAY_TASK_ID} $2 $3 $4 $5 $6 $7