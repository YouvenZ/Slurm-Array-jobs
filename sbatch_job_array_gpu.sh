#!/bin/bash
#SBATCH -p GPU48Go
#SBATCH --gres=gpu
#SBATCH -J gpu_array_test
#SBATCH -c 4
#SBATCH --mem 8000
#SBATCH --error log/gpu_array_test%A_%a.txt
#SBATCH --output log/gpu_array_test%A_%a.out
#SBATCH --array=[0-50]%5

srun singularity run --nv /data_GPU/rzeghlache/containers/carbure_image.sif python /data_GPU/rzeghlache/slurm_array/main_gpu.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --df_name study_exp_main

echo "Job $SLURM_JOB_ID with array ID $SLURM_ARRAY_TASK_ID has completed."