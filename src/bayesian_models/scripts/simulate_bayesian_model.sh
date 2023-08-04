cd ~/resource-rational-compositional-RL/

srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_compositional_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_gp_optimal_simulation.py
srun singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_compositional_gp_optimal_simulation.py

