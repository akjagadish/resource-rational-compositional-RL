conda activate pytorch-gpu

# linear-periodic-composition
python3 traina2c.py --c ${SLURM_ARRAY_TASK_ID} --prior svdo --env-name jagadish2022curriculum-v0 --no-cuda --runs 1 --entropy-loss --c-scale 10
# periodic-linear-composition
python3 traina2c.py --c ${SLURM_ARRAY_TASK_ID} --prior svdo --env-name  jagadish2022curriculum-v1  --no-cuda --runs 1 --entropy-loss --c-scale 10 
