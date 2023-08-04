cd ~/resource-rational-compositional-RL/

srun singularity exec --bind ./:/notebooks/ /u/ajagadish/resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker.py
srun singularity exec --bind ./:/notebooks/ /u/ajagadish/resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_compositional.py
srun singularity exec --bind ./:/notebooks/ /u/ajagadish/resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_gp.py
srun singularity exec --bind ./:/notebooks/ /u/ajagadish/resource-rational-compositional-RL/gpytorch-image.sif python /u/ajagadish/resource-rational-compositional-RL/src/bayesian_models/run_compositional_gp.py