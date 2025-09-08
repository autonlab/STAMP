import platform
import os

class LocalConfig:

    def __init__(self, base_dir):
        self._base_dir = base_dir

class FirstNameZFSLocalConfig(LocalConfig):
    def __init__(self):
        self._base_dir = None
        self.computer_name = platform.node()
        self.moment_models_dir = '/path/to/moment_models'
        self.benchmark_datasets_dir = '/path/to/benchmark_data'
        self.processed_data_dirs = {
            'bciciv2a': '/path/to/benchmark_data/bciciv2a/processed_inde_avg_03_50',
            'chbmit': '/path/to/benchmark_data/chbmit/processed_lmdb',
            'faced': '/path/to/benchmark_data/faced/processed',
            'isruc': '/path/to/benchmark_data/isruc/processed',
            'mumtaz': '/path/to/benchmark_data/mumtaz/processed_lmdb_75hz',
            'physio': '/path/to/benchmark_data/physio/processed_average',
            'seedv': '/path/to/benchmark_data/seedv/processed',
            'seedvig': '/path/to/benchmark_data/seedvig/processed',
            'shu': '/path/to/benchmark_data/shu/processed',
            'speech': '/path/to/benchmark_data/speech/processed',
            'stress': '/path/to//benchmark_data/stress/processed',
            'tuab': '/path/to/benchmark_data/tuab/processed',
            'tuev': '/path/to/benchmark_data/tuev/processed'
        }

def get_local_config():
    username = os.getlogin()
    computer_name = platform.node()

    if (username == 'name' and ('server' in computer_name or 'gpu' in computer_name)):
        return FirstNameZFSLocalConfig()
    else:
        return LocalConfig('/path/to/')

local_config : LocalConfig = get_local_config()
