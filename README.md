# Zero-shot compositional reinforcement learning in humans

[![Under Construction](https://img.shields.io/badge/status-under%20construction-yellow)](https://shields.io/)

Code and Data for "Zero-shot compositional reinforcement learning in humans".

<p align="center">
  <img src="figures/Experiment.png" />
</p>

## Abstract
People can easily evoke previously learned concepts, compose them, and apply the result to solve novel tasks on the first attempt. The aim of this paper is to improve our understanding of how people make such zero-shot compositional inferences in a reinforcement learning setting. To achieve this, we introduce an experimental paradigm where people learn two latent reward functions and need to compose them correctly to solve a novel task. We find that people have the capability to engage in zero-shot compositional reinforcement learning but deviate systematically from optimality. However, their mistakes are structured and can be explained by their performance in the sub-tasks leading up to the composition. Through extensive model-based analyses, we found that a meta-learned neural network model that accounts for limited computational resources best captures participants’ behaviour. Moreover, the amount of computational resources this model identified reliably quantifies how good individual participants are at zero-shot compositional reinforcement learning. Taken together, our work takes a considerable step towards studying compositional reasoning in agents – both natural and artificial – with limited computational resources.

### Data Availability
Data for experiments 1 and 2 are available in the `data` folder and can be loaded using the `pandas` library. 

### Code Availability
Code for the experiments is available in the `src` folder. 
<!-- `requirements.txt` contains the required python packages to run the code. -->

<!-- ## Requirements -->

## Instructions to run the code

#### Bayesian Models: run using singualarity container
1. Install singularity (https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps)
2. Download the si from docker hub using the following command:
```singularity pull gpytorch-image.sif docker://akjagadish/gpytorch:latest```
3. Run the models within the container. For example, to run the simulations for the mean tracker model, run the following command:
```singularity exec --bind ./:/notebooks/ /resource-rational-compositional-RL/gpytorch-image.sif python /resource-rational-compositional-RL/src/bayesian_models/run_mean_tracker_optimal_simulation.py```

#### Meta-reinforcement learning models: using anaconda
1. Install anaconda (https://docs.anaconda.com/anaconda/install/)
2. Create a new environment using the following command:
```conda create -name pytorch-gpu ```
3. Activate the environment using the following command:
```conda activate pytorch-gpu```
4. Install the required packages using the following command:
```conda install -c conda-forge python==3.9.13 numpy pytorch gym numpy pandas tqdm matplotlib tensorboard seaborn jupyterlab```
5. Train the RR-RL2 model within the conda environment.
```python3 traina2c.py --c 10000 --prior svdo --env-name jagadish2022curriculum-v0 --no-cuda --runs 1 --entropy-loss --c-scale 10```
