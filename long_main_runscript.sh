#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p unrestricted,janson,janson_cascade,shared,serial_requeue
#SBATCH -o outputs/output_%A_%a.out
#SBATCH -e errors/errors_%A_%a.err
#SBATCH -t 0-60:00 
#SBATCH --array=1-200


module purge
module load Anaconda3/2020.11
python testing_driver.py $1 ${SLURM_ARRAY_TASK_ID} $2 $3 $4 $5 $6 $7 $8