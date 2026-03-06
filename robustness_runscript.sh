#!/bin/bash
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -p # Partition to submit to
#SBATCH -o outputs/output_%A_%a.out
#SBATCH -e errors/errors_%A_%a.err
#SBATCH -t 0-02:00 
#SBATCH --array=0-971

module load python/3.10.13-fasrc01
mamba run -n y<conda-environment>h python3 robustness_check.py ${SLURM_ARRAY_TASK_ID}
