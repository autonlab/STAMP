from stamp.experiments import (
    run_full_experiment
)
from stamp.local import get_local_config
import os

import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

local_config = get_local_config()

dataset_names = ['bciciv2a']

exp_names = [
'MOMENT-1-large_LoRA-r64-a32-dr0.05_nrs3_ne15_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_ls0.1gcT_inorm_tdr1.0'
]

experiments_dir = '/path/to/stamp/experiments'
device = 'cuda:3'

for dataset_name in dataset_names:
    for exp_name in exp_names:

        if os.path.exists(os.path.join(experiments_dir, dataset_name, exp_name, 'figures')):
            print(f"Experiment {exp_name} for dataset {dataset_name} already has a /figures folder. Skipping...")
            continue
        run_full_experiment(dataset_name, exp_name, experiments_dir, device, exp_type='lora')