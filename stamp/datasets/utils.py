import torch
import pandas as pd
from momentfm import MOMENTPipeline
from tsfm_public.models.tspulse import TSPulseForReconstruction
from chronos import ChronosPipeline

dataset_params = {
    'faced': {
        'orig_sampling_rate': 250,
        'n_spatial_channels': 32,
        'n_temporal_channels': 10,
        'n_samples': 10332,
        'n_classes': 9

    },
    'seedv': {
        'orig_sampling_rate': 1000,
        'n_spatial_channels': 62,
        'n_temporal_channels': 1,
        'n_samples': 117744,
        'n_classes': 5

    },
    'physio': {
        'orig_sampling_rate': 160,
        'n_spatial_channels': 64,
        'n_temporal_channels': 4,
        'n_samples': 9837,
        'n_classes': 4

    },
    'shu': {
        'orig_sampling_rate': 250,
        'n_spatial_channels': 32,
        'n_temporal_channels': 4,
        'n_samples': 11988,
        'n_classes': 1
    },
    'isruc': {
        'orig_sampling_rate': 200,
        'n_spatial_channels': 6,
        'n_temporal_channels': 30,
        'n_samples': 89240,
        'n_classes': 5
    },
    'chbmit': {
        'orig_sampling_rate': 256,
        'n_spatial_channels': 16,
        'n_temporal_channels': 10,
        'n_samples': 326993,
        'n_classes': 1
    },
    'speech': {
        'orig_sampling_rate': 256,
        'n_spatial_channels': 64,
        'n_temporal_channels': 3,
        'n_samples': 6000,
        'n_classes': 5
    },
    'mumtaz': {
        'orig_sampling_rate': 256,
        'n_spatial_channels': 19,
        'n_temporal_channels': 5,
        'n_samples': 7143,
        'n_classes': 1
    },
    'seedvig': {
        'orig_sampling_rate': 200,
        'n_spatial_channels': 17,
        'n_temporal_channels': 8,
        'n_samples': 20355,
        'n_classes': 'regression'
    },
    'stress': {
        'orig_sampling_rate': 500,
        'n_spatial_channels': 20,
        'n_temporal_channels': 5,
        'n_samples': 1707,
        'n_classes': 1
    },
    'tuev': {
        'orig_sampling_rate': 250,
        'n_spatial_channels': 16,
        'n_temporal_channels': 5,
        'n_samples': 112491,
        'n_classes': 6
    },
    'tuab': {
        'orig_sampling_rate': 250,
        'n_spatial_channels': 16,
        'n_temporal_channels': 10,
        'n_samples': 409455,
        'n_classes': 1
    },
    'bciciv2a': {
        'orig_sampling_rate': 250,
        'n_spatial_channels': 22,
        'n_temporal_channels': 4,
        'n_samples': 5088,
        'n_classes': 4
    },
}

def get_embeddings_dir(dataset_name, embedding_model_name, datasets_dir):
    return datasets_dir + f'/{dataset_name}/{embedding_model_name}'

def get_dataset_params(dataset_name):
    """
    Get the dataset parameters for the given dataset name.
    """
    if dataset_name not in dataset_params:
        raise ValueError(f"Dataset {dataset_name} not found in dataset_params. Available datasets: {list(dataset_params.keys())}")

    return dataset_params[dataset_name]

def get_problem_type(n_classes):
    if n_classes == 'regression':
        problem_type = 'regression'
    else:
        if n_classes == 1:
            problem_type = 'binary'
        elif n_classes > 1:
            problem_type = 'multiclass'
        else:
            raise ValueError('n_classes cannot be negative.')

    return problem_type

def get_monitor_metric(problem_type):
    if problem_type == 'binary':
        monitor_metric = 'val_roc_auc'
    elif problem_type == 'multiclass':
        monitor_metric = 'val_cohen_kappa'
    elif problem_type == 'regression':
        monitor_metric = 'val_mse'
    else:
        raise ValueError(f'Unknown classification type: {problem_type}')

    return monitor_metric

def verify_data_integrity(features_df, labels_df, n_temporal_channels, n_spatial_channels):
    """
    Verify that the trial IDs and labels are consistent.
    """
    print("🔍 Verifying data integrity...")

    # Check that each original trial has correct number of embeddings
    trial_counts = features_df['sample_key'].value_counts()
    expected_count = n_temporal_channels * n_spatial_channels  # n_temporal_channels * n_spatial_channels

    incorrect_counts = trial_counts[trial_counts != expected_count]
    if len(incorrect_counts) > 0:
        print(f"Found {len(incorrect_counts)} samples with incorrect embedding counts:")
        print(incorrect_counts.head())
        print(f'Expected {n_temporal_channels} * {n_spatial_channels} = {expected_count} embeddings per sample.')
        return False

    # Check that labels are consistent within each trial
    label_consistency = features_df.groupby('sample_key')['original_label'].nunique()
    inconsistent_labels = label_consistency[label_consistency > 1]
    if len(inconsistent_labels) > 0:
        print(f"Found {len(inconsistent_labels)} trials with inconsistent labels")
        return False

    # Check that labels DataFrame matches features DataFrame
    features_labels = features_df.groupby('sample_key')['original_label'].first().reset_index()
    features_labels.columns = ['trial_id', 'label']

    merged = pd.merge(labels_df, features_labels, on=['trial_id', 'label'], how='outer', indicator=True)
    mismatches = merged[merged['_merge'] != 'both']
    if len(mismatches) > 0:
        print(f"Found {len(mismatches)} label mismatches between DataFrames")
        return False

    print("All data integrity checks passed!")
    return True

def verify_master_df(master_df, n_temporal_channels, n_spatial_channels):
    sample_counts = master_df.index.get_level_values('sample_key').value_counts()
    expected_count = n_temporal_channels * n_spatial_channels
    assert (sample_counts == expected_count).all(), f'There are samples which do not have the expected number of embeddings. Each sample should have {expected_count} embeddings.'

    return len(sample_counts)

def verify_all_master_dfs(master_dfs, n_temporal_channels, n_spatial_channels, n_samples):
    total_samples = 0
    for master_df in master_dfs:
        total_samples += verify_master_df(master_df=master_df, n_temporal_channels=n_temporal_channels, n_spatial_channels=n_spatial_channels)

    assert total_samples == n_samples, f'The number of samples across the master dataframes, {total_samples}, does not match the expected number of samples (from the CBraMod paper) which is {n_samples} samples.'

def verify_data_loader(data_loader, n_samples):
    train_len = len(data_loader['train'].dataset)
    val_len = len(data_loader['val'].dataset)
    test_len = len(data_loader['test'].dataset)

    total_len = train_len + val_len + test_len

    assert total_len == n_samples, f'The number of samples stored in the data loader, {total_len}, does not match the expected number of samples (from the CbraMod paper) which is {n_samples} samples.'

def load_moment_model(moment_model_name, moment_models_dir, **kwargs):
    model = MOMENTPipeline.from_pretrained(
        f"AutonLab/{moment_model_name}",
        model_kwargs={
            "task_name": "embedding"
        },
        # Give local dir
        # to avoid downloading the model every time
        cache_dir = moment_models_dir,
        local_files_only=True
    )

    model.init(); # NOTE: IMPORTANT!!! Otherwise, the model will not be initialized properly.

    return model

def load_tspulse_model(model_name, tspulse_models_dir, **kwargs):
    # This is adapted from https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/tspulse_search_simple_example.ipynb
    model = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-hybrid-dualhead-512-p8-r1",
        num_input_channels=1,
        mask_type="user",
        cache_dir=tspulse_models_dir,
        local_files_only=True
    )

    return model

def load_chronos_model(model_name, chronos_models_dir, device, **kwargs):
    model = ChronosPipeline.from_pretrained(
        f"amazon/{model_name}",
        dtype=torch.bfloat16,
        device_map=device,
        cache_dir=chronos_models_dir,
        local_files_only=True
    )

    # Move all tensor attributes in tokenizer to device
    for attr_name in dir(model.tokenizer):
        if not attr_name.startswith('_'):
            attr = getattr(model.tokenizer, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(model.tokenizer, attr_name, attr.to(device))

    return model
    
def get_simple_gpu_info(available_gpus):
    """Get simple GPU info: just names and memory"""

    gpu_info = []

    for gpu_id in available_gpus:
        gpu_id = int(gpu_id)
        if int(gpu_id) < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            gpu_info.append({
                'gpu_id': gpu_id,
                'name': props.name,
                'memory_gb': round(props.total_memory / (1024**3), 1)
            })

    return gpu_info

def get_shuffled_sample_keys(dataset, dataloader):
    """Extract sample keys in DataLoader order without loading data"""
    # Get the indices in the order the sampler will yield them
    if hasattr(dataloader.sampler, '__iter__'):
        sampler_indices = list(dataloader.sampler)
    else:
        # Fallback for custom samplers
        sampler_indices = list(range(len(dataset)))
        if dataloader.shuffle:
            import random
            random.shuffle(sampler_indices)

    # Map indices to sample keys
    return [dataset.keys[idx] for idx in sampler_indices]