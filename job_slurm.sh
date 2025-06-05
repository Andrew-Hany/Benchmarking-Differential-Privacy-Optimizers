#!/bin/bash
#SBATCH --job-name=hyperparam_tuning
#SBATCH --account=project_2003275
#SBATCH --error=errorlog/error_%A_%a.txt
#SBATCH --output=log/output_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-47
#SBATCH --partition=gpusmall  

mkdir -p errorlog
mkdir -p log
# Activate your virtual environment
source /scratch/project_2003275/Andrew_temp/.venv/bin/activate


# Run the main script and log output
python /scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/main_slurm.py 
