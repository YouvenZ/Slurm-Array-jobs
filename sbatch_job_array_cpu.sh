#!/bin/bash
#SBATCH -p Serveurs-CPU ## Need to be changed based on your cluster
#SBATCH -J cpu_array_test
#SBATCH -c 4
#SBATCH --mem 8000
#SBATCH --error log/cpu_array_test%A_%a.txt
#SBATCH --output log/cpu_array_test%A_%a.out
#SBATCH --array=[0-49]%25

srun singularity run /data_GPU/rzeghlache/containers/carbure_image.sif python /data_GPU/rzeghlache/slurm_array/main_cpu.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --df_name study_exp_main

echo "Job $SLURM_JOB_ID with array ID $SLURM_ARRAY_TASK_ID has completed."