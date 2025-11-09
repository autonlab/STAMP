from CBraMod.datasets import lmdb_pickle_dataset, lmdb_np_dataset
from stamp.datasets.utils import load_moment_model, load_chronos_model, load_tspulse_model
from stamp.embeddings import build_embeddings
from stamp.local import get_local_config
from types import SimpleNamespace

local_config = get_local_config()

if __name__ == '__main__':

    batch_size = 4 # NOTE: Each sample isn't an individual time series, but a (spatial_channels x temporal channels) matrix.
    available_gpus = [0,1,2,3] # List of available GPU IDs
    embedding_model_name = 'MOMENT-1-large' # Options: ['MOMENT-1-small', 'MOMENT-1-base', 'MOMENT-1-large', 'chronos-t5-large', 'TSPulse']
    model_load_fn = load_moment_model # Options: [load_moment_model, load_chronos_model, load_tspulse_model]
    model_dir = local_config.moment_models_dir # Options: [local_config.moment_models_dir, local_config.chronos_models_dir, local_config.tspulse_models_dir]

    tmp_dir = '/path/to/tmp_chunks' # Temporary directory to store intermediate chunk files
    dataset_name = 'shu'
    output_path = f'/path/to/benchmark_data/{dataset_name}/{embedding_model_name}' # Directory to save the generated embeddings
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
            reshape_data=True,
            orig_seq_len=200,
            embedding_model_name=embedding_model_name
    )

    dataset = lmdb_pickle_dataset.LoadDataset(params)

    build_embeddings(
        dataset=dataset,
        available_gpus=available_gpus,
        chunk_size=400,
        temp_chunks_dir=tmp_dir,
        model_name=embedding_model_name,
        model_dir=model_dir,
        model_load_fn=model_load_fn,
        use_amp=False,
        output_path=output_path,
        map_size_per_mode=map_size_per_mode
    )