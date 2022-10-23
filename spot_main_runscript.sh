#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p unrestricted,shared,serial_requeue
#SBATCH -o outputs/output_%A.out
#SBATCH -e errors/errors_%A.err
#SBATCH -t 0-8:00 
#SBATCH --array=1-2000


module purge
module load Anaconda3/2020.11
python testing_driver.py $1 ${SLURM_ARRAY_TASK_ID} $2 $3 $4 $5 $6 $7 $8