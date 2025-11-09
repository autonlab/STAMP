from stamp.experiments import (
    run_full_experiment
)
from stamp.local import get_local_config

local_config = get_local_config()

dataset_names = ['tuev']

exp_names = [
'CBraMod_experiment_mlr-T_tdr0.3'
]

experiments_dir = '/path/to/cbramod/experiments'
device = '3'

for dataset_name in dataset_names:
    for exp_name in exp_names:
        run_full_experiment(dataset_name, exp_name, experiments_dir, device, exp_type='cbramod')