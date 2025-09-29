#!/bin/bash
#SBATCH -p Serveurs-CPU
#SBATCH -J Create_exp
#SBATCH -c 4
#SBATCH --mem 8000
#SBATCH --error log/Create_exp%j.txt
#SBATCH --output log/Create_exp%j.out

srun singularity run /data_GPU/rzeghlache/containers/carbure_image.sif python /data_GPU/rzeghlache/slurm_array/parser.py
srun singularity run /data_GPU/rzeghlache/containers/carbure_image.sif python /data_GPU/rzeghlache/slurm_array/create_exp.py
echo "Job create csv has completed."
