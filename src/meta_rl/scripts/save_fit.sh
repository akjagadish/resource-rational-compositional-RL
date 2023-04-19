#!/bin/bash -l
#SBATCH -o ./logs/tjob.out.%A_%a
#SBATCH -e ./logs/tjob.err.%A_%a
#SBATCH --job-name=exploration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=8

cd ~/RL3NeurIPS/

module purge
conda activate pytorch-gpu

# fit policy parameters to all subtasks
python3 save_fits.py.py --full --changepoint --entropy
python3 save_fits.py.py --full --entropy

# fit policy parameters only to last subtask
python3 save_fits.py.py --changepoint --entropy
python3 save_fits.py.py --entropy