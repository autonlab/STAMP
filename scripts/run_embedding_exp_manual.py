import sys
sys.path.append('../../STAMP')
sys.path.append('../CBraMod')
from stamp.experiments import (
    run_full_experiment
)
from stamp.local import get_local_config
import os

local_config = get_local_config()

dataset_names = ['shu']

exp_names = [
# 'MOMENT-1-large_nrs3_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tbasic-nl8-dff256-dr0.3_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_ls0.1gcT_inorm_tdr1.0',
# 'MOMENT-1-large_nrs3_ne50_D128_ip-full-dr0.3_pe-basicNST_tf-tbasic-A8-nl8-dff256-dr0.3_mhap-A4-dr0.3-Q8-qcweighted_sum_ls0.1gcT_inorm_tdr1.0',
'MOMENT-1-large_nrs3_ne50_D128_ip-full-dr0.3_pe-basicNST_tf-tcriss_cross-A8-nl8-dff256-dr0.3_mhap-A4-dr0.3-Q8-qcweighted_sum_ls0.1gcT_inorm_tdr1.0'
]

experiments_dir = '/path/to/stamp/experiments'
device = 'cuda:3'

for dataset_name in dataset_names:
    for exp_name in exp_names:

        if os.path.exists(os.path.join(experiments_dir, dataset_name, exp_name, 'figures')):
            print(f"Experiment {exp_name} for dataset {dataset_name} already has a /figures folder. Skipping...")
            continue
        run_full_experiment(dataset_name, exp_name, experiments_dir, device, exp_type='embedding')