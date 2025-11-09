from stamp.experiments import (
    run_full_experiment
)
from stamp.local import get_local_config
import os

local_config = get_local_config()

dataset_names = ['stress']

exp_names = [
'MOMENT-1-large_nrs3_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0'
]

experiments_dir = local_config.tsfm_experiments_dir
device = 'cuda:3'

for dataset_name in dataset_names:
    for exp_name in exp_names:

        if os.path.exists(os.path.join(experiments_dir, dataset_name, exp_name, 'figures')):
            print(f"Experiment {exp_name} for dataset {dataset_name} already has a /figures folder. Skipping...")
            continue
        run_full_experiment(dataset_name, exp_name, experiments_dir, device, exp_type='embedding')