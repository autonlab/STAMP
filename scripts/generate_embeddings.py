import sys
sys.path.append('../CBraMod')
sys.path.append('../../STAMP')

from CBraMod.datasets import tuab_dataset
from CBraMod.datasets import lmdb_pickle_dataset, lmdb_np_dataset
from CBraMod.datasets.utils import *
from CBraMod.datasets.distributed_utils import *
from stamp.local import get_local_config
from types import SimpleNamespace

local_config = get_local_config()

if __name__ == '__main__':

    batch_size = 4
    available_gpus = [1]
    moment_model_name = 'MOMENT-1-large'

    tmp_dir = '/path/to/tmp_chunks'
    dataset_name = 'shu'
    output_path = f'/path/to/benchmark_data/{dataset_name}/{moment_model_name}'
    processed_data_dirs = local_config.processed_data_dirs
    processed_data_dir = processed_data_dirs[dataset_name]

    map_size_per_mode = {
        'train': 10 * 1024**3,
        'val': 7 * 1024**3,
        'test': 7 * 1024**3
    }
    params = SimpleNamespace(
            dataset_name=dataset_name,
            dataset_dir=processed_data_dir,
            batch_size=batch_size,
            return_mask=True,
            pad_to_len=512,
            reshape_data=True,  # Set to True to reshape data to (batch_size * n_channels * n_seconds, seq_len)
            orig_seq_len=200
    )

    dataset = lmdb_pickle_dataset.LoadDataset(params)
    # dataset = lmdb_np_dataset.LoadDataset(params)

    process_distributed_all_modes(
        dataset=dataset,
        available_gpus=available_gpus,
        chunk_size=400,
        temp_chunks_dir=tmp_dir,
        model_name=moment_model_name,
        model_dir=local_config.moment_models_dir,
        model_load_fn=load_moment_model,
        use_amp=False,
        output_path=output_path,
        map_size_per_mode=map_size_per_mode
    )
    # TUAB Dataset
    # tuab_dataset_dir='/path/to/benchmark_data/TUAB/v3.0.1/edf/process_refine'
    # tuab_embeddings_dir=f'/path/to/benchmark_data/TUAB/v3.0.1/{moment_model_name}_embeddings'
    # tuab_params = SimpleNamespace(
    #     dataset_dir=tuab_dataset_dir,
    #     batch_size=batch_size,
    #     return_mask=True,
    #     pad_to_len=512,
    #     reshape_data=True,  # Set to True to reshape data to (batch_size * n_channels * n_seconds, seq_len)
    # )

    # tuab_data = tuab_dataset.LoadDataset(tuab_params)

    # process_distributed(
    #     dataset=tuab_data,
    #     mode='train',
    #     available_gpus=available_gpus,
    #     chunk_size=2,
    #     embeddings_dir=tuab_embeddings_dir,
    #     model_name=moment_model_name,
    #     model_dir=local_config.moment_models_dir,
    #     model_load_fn=load_moment_model,
    #     use_amp=False,
    #     merge_results=True
    # )