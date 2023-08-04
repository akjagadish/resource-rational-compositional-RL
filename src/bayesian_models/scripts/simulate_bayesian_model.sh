#!/bin/bash -l

#SBATCH -o /eris/scratch/ajagadish/logs/tjob.out.%A_%a
#SBATCH -e /eris/scratch/ajagadish/logs/tjob.err.%A_%a

# --- resource specification (which resources for how long) ---
#SBATCH --partition=s.eris
#SBATCH --job-name=MTrackerSim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# --SBATCH --nodes=1
# --SBATCH --ntasks-per-node=1
# --SBATCH --ntasks=1
# --SBATCH --ntasks-per-node=2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# if [ $SLURM_CPUS_PER_TASK="" ]
# then
#     echo "export OMP_NUM_THREADS=1"
# else
#     echo "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}"
# fi

# --- start from a clean state and load necessary environment modules ---
module purge

cd ~/resource-rational-compositional-RL/

srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_compositional_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_gp_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_compositional_gp_optimal_simulation.py

