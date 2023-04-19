#!/bin/bash -l
#SBATCH -o ./logs/tjob.out.%A_%a
#SBATCH -e ./logs/tjob.err.%A_%a
#SBATCH --job-name=RL3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8

cd ~/RL3NeurIPS/

module purge
#module load gcc/10
#module load Anaconda3 #/3/2020.02
conda activate pytorch-gpu
# all subtasks
#python simulate.py.py --full --entropy --prior svdo --num-episodes 400
#python simulate.py.py --full --entropy --prior svdo --num-episodes 400 --changepoint
#python simulate.py.py --full --entropy --prior svdo --num-episodes 400 --changepoint --env-name jagadish2022curriculum-v1

# last subtasks
# python simulate.py.py --entropy --prior svdo --num-episodes 400
#python simulate.py.py --entropy --prior svdo --num-episodes 400 --changepoint
#python simulate.py.py --entropy --prior svdo --num-episodes 400 --changepoint --env-name jagadish2022curriculum-v1

## NUM EPISODES = 400 TO KEEP IT SIMILAR TO GRAMMAR COMPOSITIONS

## per trial
# python simulate.py.py --entropy --prior svdo --num-episodes 400 --per-trial 0
python simulate.py.py --entropy --prior svdo --num-episodes 400 --changepoint --per-trial 0
#python simulate.py.py --entropy --prior svdo --num-episodes 400 --changepoint --per-trial 0 --env-name jagadish2022curriculum-v1