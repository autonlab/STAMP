import numpy as np
import subprocess
import datetime
import torch
import random
import time

def set_commit_hash_and_run_date_in_config(
    exp_config
    ):
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f'{e=} The directory is not a git repo so the commit hash could not be retrieved.')
        raise

    exp_config['commit_hash'] = commit_hash
    exp_config['run_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_run_time(fn):
    def inner(*args, **kwargs):
        start = time.time()
        fn_output = fn(*args, **kwargs)
        print((time.time() - start))
        return fn_output
    return inner

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

