import os
import time
import json
import pandas as pd
from types import SimpleNamespace
import numpy as np
import random
import matplotlib.pyplot as plt
from shutil import rmtree as shutil_rmtree
from CBraMod.datasets import lmdb_embedding_dataset, lmdb_np_dataset, lmdb_pickle_dataset
from stamp.datasets.utils import (verify_data_loader, 
                                    get_simple_gpu_info, get_dataset_params, get_problem_type,
                                    get_monitor_metric, get_embeddings_dir)
from stamp.common import set_commit_hash_and_run_date_in_config, setup_seed
from stamp.modeling import create_modeling_approach
from stamp.modeling.utils import calculate_binary_performance_metrics, calculate_multiclass_performance_metrics
from stamp.local import get_local_config

local_config = get_local_config()

def create_experiment(
    exp_dir:str,
    exp_config:dict
    ):
    """Creates a new experiment by creating a new folder
    within the experiments directory and saving a JSON
    containing the exp_config for the experiment.

    Parameters
    ----------
    experiments_dir : str
        The path to the experiments directory.

    exp_config : dict
        The configuration dictionary for the experiment.
    """
    # Create the config for the experiment
    exp_name = exp_config['exp_name']

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_dir + f"/{exp_name}/", exist_ok=False)

    # Save experiment config
    with open(exp_dir + f"/{exp_name}/exp_config.json", 'w') as file:
        json.dump(exp_config, file, indent=4)

    print(f"Created Experiment {exp_name}...")
    return

def load_experiment_config(
    curr_exp_dir:str
    )->dict:
    """Load an experiment's config dictionary.

    Parameters
    ----------


    Returns
    -------
    dict
        The config dictionary for the experiment.
    """
    # Extract experiment config
    with open(curr_exp_dir + '/exp_config.json', 'r') as file:
        exp_config = json.load(file)

    return exp_config

def run_full_experiment(
    dataset_name,
    exp_name:str,
    experiments_dir:str,
    device:str,
    exp_type,
    exp_config:dict=None,
    )->dict[int,pd.DataFrame]:
    """Runs a full experiment.

    Parameters
    ----------
    exp_name : str
        The name of the experiment that should be loaded.

    experiments_dir : str
        The path to the experiments directory.

    exp_config : dict
        The exp_config dictionary for the experiment.
    """

    curr_exp_dir = experiments_dir + f'/{dataset_name}/{exp_name}'

    # Load the experiment config
    if exp_config is None:
        exp_config = load_experiment_config(curr_exp_dir)
    
    if exp_type == 'embedding':
        embeddings_dir = get_embeddings_dir(dataset_name=dataset_name, embedding_model_name=exp_config['embedding_model_name'], datasets_dir=local_config.datasets_dir)
        processed_data_dir = None
    elif exp_type =='cbramod' or exp_type =='eeg_conformer' or exp_type=='lora':
        processed_data_dir = local_config.processed_data_dirs[dataset_name]
        embeddings_dir = None
    else:
        raise ValueError(f"exp_type {exp_type} not recognized")
    
    dataset_params = get_dataset_params(dataset_name=dataset_name)
    n_classes = dataset_params['n_classes']
    problem_type = get_problem_type(n_classes)
    monitor_metric = get_monitor_metric(problem_type)

    exp_config = update_dataset_specific_params_in_exp_config(
        exp_config=exp_config,
        embeddings_dir=embeddings_dir,
        processed_data_dir=processed_data_dir,
        problem_type=problem_type,
        n_classes=n_classes,
        monitor_metric=monitor_metric,
        dataset_name=dataset_name,
        exp_name=exp_name,
        exp_type=exp_type
    )

    print(exp_config.keys())

    os.makedirs(curr_exp_dir + f'/figures', exist_ok=False)

    # Check if checkpoint dir is in checkpointing_params, if so create it
    if exp_config['modeling_approach_config']['params'].get('checkpointing_params') is not None:
        checkpoint_dir = exp_config['modeling_approach_config']['params']['checkpointing_params'].get('checkpoint_dir')
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=False)

    main_random_seed = 42

    # Handle the random seeds
    n_random_seeds = exp_config['n_random_seeds']
    assert n_random_seeds < 1000, "n_random_seeds must be less than 1000 to ensure unique random seeds."
    random_seeds = random.Random(main_random_seed).sample(range(1000), n_random_seeds)
    exp_config['random_seeds'] = random_seeds

    set_commit_hash_and_run_date_in_config(exp_config=exp_config)

    # Set device
    if device != 'cpu':
        # Get the GPU ID from 'cuda:X'
        gpu_id = device.split(':')[-1]
        gpu_info = get_simple_gpu_info([gpu_id])
        exp_config['gpu_info'] = gpu_info

    modeling_approach_config = exp_config['modeling_approach_config']
    exp_config['modeling_approach_config']['params']['device'] = device

    print(f'Starting experiment...')
    main_start = time.time()
    training_run_time_per_seed = {}
    testing_run_time_per_seed = {}
    pred_df_per_seed = {}
    extra_info_per_seed = {}
    performance_metrics_per_seed = {}
    for random_seed in random_seeds:
        setup_seed(random_seed)

        if exp_type == 'embedding':
            data_loader, loaddataset = build_embedding_data_loader(
                dataset_name=dataset_name,
                embeddings_dir=embeddings_dir,
                batch_size=modeling_approach_config['params']['train_batch_size'],
                tdr=exp_config['training_data_ratio'],
                seed=random_seed,
                temporal_channel_selection=exp_config.get('temporal_channel_selection', None)
            )
        elif exp_type == 'cbramod':
            data_loader, loaddataset = build_cbramod_data_loader(
                dataset_name=dataset_name,
                processed_data_dir=processed_data_dir,
                batch_size=modeling_approach_config['params']['train_batch_size'],
                tdr=exp_config['training_data_ratio'],
                seed=random_seed
            )
        elif exp_type == 'eeg_conformer':
            data_loader, loaddataset = build_eeg_conformer_data_loader(
                dataset_name=dataset_name,
                processed_data_dir=processed_data_dir,
                batch_size=modeling_approach_config['params']['train_batch_size'],
                tdr=exp_config['training_data_ratio'],
                seed=random_seed
            )
        elif exp_type == 'lora':
            data_loader, loaddataset = build_lora_data_loader(
                dataset_name=dataset_name,
                processed_data_dir=processed_data_dir,
                batch_size=modeling_approach_config['params']['train_batch_size'],
                tdr=exp_config['training_data_ratio'],
                seed=random_seed
            )
        else:
            raise ValueError()
        
        exp_config['modeling_approach_config']['params']['n_temporal_channels'] = loaddataset.dataset_params['n_temporal_channels']
        exp_config['modeling_approach_config']['params']['n_spatial_channels'] = loaddataset.dataset_params['n_spatial_channels']
        exp_config['modeling_approach_config']['params']['n_samples'] = loaddataset.dataset_params['n_samples']

        print(f'Starting training for random seed {random_seed}...')
        train_start = time.time()
        modeling_approach = create_modeling_approach(modeling_approach_config=modeling_approach_config)
        modeling_approach.random_seed = random_seed
        modeling_approach.train(
            train_data_loader=data_loader['train'],
            val_data_loader=data_loader['val']
        )

        training_run_time = time.time() - train_start
        print(f'Finished training for random seed {random_seed}...{training_run_time}')

        test_start = time.time()
        print(f'Starting testing for random seed {random_seed}...')
        pred_df, extra_info = modeling_approach.predict(
            test_data_loader=data_loader['test']
        )
        testing_run_time = time.time() - test_start
        print(f'Finished testing for random seed {random_seed}...{testing_run_time}')

        plot_all_train_val_curves(
            extra_info=extra_info,
            exp_dir=curr_exp_dir,
            seed=random_seed,
            problem_type=exp_config['modeling_approach_config']['params']['problem_type'],
        )

        plt.close('all')
        # Calculate performance metrics for test set
        prob_df = extra_info['prob_df']
        truths = extra_info['test_labels']
        probs = prob_df.values
        preds = pred_df.values

        performance_metrics = calculate_performance_metrics(
            problem_type=exp_config['modeling_approach_config']['params']['problem_type'],
            truths=truths,
            probs=probs,
            preds=preds
        )

        training_run_time_per_seed[random_seed] = training_run_time
        testing_run_time_per_seed[random_seed] = testing_run_time
        pred_df_per_seed[random_seed] = pred_df
        extra_info_per_seed[random_seed] = extra_info
        performance_metrics_per_seed[random_seed] = performance_metrics

    total_params = sum(p.numel() for p in modeling_approach.model.parameters() if p.requires_grad)

    # Calculate mean and std of performance metrics
    mean_performance_metrics, std_performance_metrics = calculate_mean_and_std_performance_metrics(
        performance_metrics_per_seed=performance_metrics_per_seed
    )
    main_run_time = time.time() - main_start
    print(f'Finished experiment {exp_name}...{main_run_time}')

    print(f'Experiment {exp_name} finished with the following performance metrics:')
    for metric, value in mean_performance_metrics.items():
        if metric == 'confusion_matrix':
            continue
        print(f'{metric}: {value:.4f} ± {std_performance_metrics[metric]:.4f}')

    exp_config['training_run_time_per_seed'] = training_run_time_per_seed
    exp_config['testing_run_time_per_seed'] = testing_run_time_per_seed
    exp_config['main_run_time'] = main_run_time
    exp_config['total_parameters'] = total_params
    exp_config['total_flops'] = modeling_approach.total_flops if hasattr(modeling_approach, 'total_flops') else None
    # Save the results
    if 'tmp' not in experiments_dir:
        exp_config_file_path = curr_exp_dir + '/exp_config.json'
        with open(exp_config_file_path, 'w') as file:
            json.dump(exp_config, file, indent=4)

        os.makedirs(curr_exp_dir + '/results', exist_ok=False)
        pd.to_pickle(pred_df_per_seed, curr_exp_dir + '/results/pred_df_per_seed.pkl')
        pd.to_pickle(extra_info_per_seed, curr_exp_dir + '/results/extra_info_per_seed.pkl')
        pd.to_pickle(performance_metrics_per_seed, curr_exp_dir + '/results/performance_metrics_per_seed.pkl')
        pd.to_pickle(mean_performance_metrics, curr_exp_dir + '/results/mean_performance_metrics.pkl')
        pd.to_pickle(std_performance_metrics, curr_exp_dir + '/results/std_performance_metrics.pkl')

    shutil_rmtree(exp_config['modeling_approach_config']['params']['early_stopping_params']['tmp_dir'])

def build_embedding_data_loader(
    dataset_name,
    embeddings_dir,
    batch_size,
    tdr,
    seed,
    temporal_channel_selection
    ):
    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=embeddings_dir,
        batch_size=batch_size,
        tdr=tdr,
        seed=seed,
        temporal_channel_selection=temporal_channel_selection,
    )

    loaddataset = lmdb_embedding_dataset.LoadDataset(params)
    data_loader = loaddataset.get_data_loader()

    if dataset_name != 'tuev' and tdr == 1.0:
        verify_data_loader(data_loader=data_loader, n_samples=loaddataset.dataset_params['n_samples'])

    return data_loader, loaddataset

def build_cbramod_data_loader(
    dataset_name,
    processed_data_dir,
    batch_size,
    tdr,
    seed
    ):
    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=processed_data_dir,
        batch_size=batch_size,
        tdr=tdr,
        seed=seed,
        reshape_data=False,
        pad_to_len=None,
        return_mask=False,
        orig_seq_len=200
    )

    pickle_datasets = [
            'bciciv2a',
            'faced',
            'mumtaz',
            'physio',
            'seedv',
            'seedvig',
            'shu',
            'speech',
            'stress'
        ]
    
    if params.dataset_name in pickle_datasets:
        loaddataset = lmdb_pickle_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()
    else:
        loaddataset = lmdb_np_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()

    if dataset_name != 'tuev' and tdr == 1.0:
        verify_data_loader(data_loader=data_loader, n_samples=loaddataset.dataset_params['n_samples'])

    return data_loader, loaddataset

def build_eeg_conformer_data_loader(
    dataset_name,
    processed_data_dir,
    batch_size,
    tdr,
    seed
    ):
    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=processed_data_dir,
        batch_size=batch_size,
        tdr=tdr,
        seed=seed,
        reshape_data=False,
        pad_to_len=None,
        return_mask=False,
        orig_seq_len=200,
        embedding_model_name='eeg_conformer' # Ensures the correct reshaping is done
    )

    pickle_datasets = [
            'bciciv2a',
            'faced',
            'mumtaz',
            'physio',
            'seedv',
            'seedvig',
            'shu',
            'speech',
            'stress'
        ]
    
    if params.dataset_name in pickle_datasets:
        loaddataset = lmdb_pickle_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()
    else:
        loaddataset = lmdb_np_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()
        
    if dataset_name != 'tuev' and tdr == 1.0:
        verify_data_loader(data_loader=data_loader, n_samples=loaddataset.dataset_params['n_samples'])

    return data_loader, loaddataset

def build_lora_data_loader(
    dataset_name,
    processed_data_dir,
    batch_size,
    tdr,
    seed
    ):
    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=processed_data_dir,
        batch_size=batch_size,
        tdr=tdr,
        seed=seed,
        reshape_data=True,
        pad_to_len=512, # Assumes moment-1-large is being used
        return_mask=True,
        orig_seq_len=200
    )

    pickle_datasets = [
            'bciciv2a',
            'faced',
            'mumtaz',
            'physio',
            'seedv',
            'seedvig',
            'shu',
            'speech',
            'stress'
        ]
    
    if params.dataset_name in pickle_datasets:
        loaddataset = lmdb_pickle_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()
    else:
        loaddataset = lmdb_np_dataset.LoadDataset(params)
        data_loader = loaddataset.get_data_loader()
        
    if dataset_name != 'tuev' and tdr == 1.0:
        verify_data_loader(data_loader=data_loader, n_samples=loaddataset.dataset_params['n_samples'])

    return data_loader, loaddataset
        
def plot_all_train_val_curves(
    extra_info:dict,
    exp_dir:str,
    seed:int,
    problem_type:str
):
    exp_figures_dir = exp_dir + '/figures'

    plot_train_val_curves(
        train_values=extra_info['train_main_losses'],
        val_values=extra_info['val_main_losses'],
        label='Total Loss',
        title=f'Train and Validation Loss, Seed Index: {seed}',
        best_epoch=extra_info['best_epoch'],
        exp_figures_dir=exp_figures_dir,
        seed=seed
    )

    plot_train_val_curves(
        train_values=extra_info['train_balanced_acc_list'],
        val_values=extra_info['val_balanced_acc_list'],
        label='Balanced Accuracy',
        title=f'Train and Validation Balanced Accuracy, Seed Index: {seed}',
        best_epoch=extra_info['best_epoch'],
        exp_figures_dir=exp_figures_dir,
        seed=seed
    )

    if problem_type == 'binary':
        plot_train_val_curves(
            train_values=extra_info['train_pr_auc_list'],
            val_values=extra_info['val_pr_auc_list'],
            label='AUC-PR',
            title=f'Train and Validation AUC-PR, Seed Index: {seed}',
            best_epoch=extra_info['best_epoch'],
            exp_figures_dir=exp_figures_dir,
            seed=seed
        )

        plot_train_val_curves(
            train_values=extra_info['train_roc_auc_list'],
            val_values=extra_info['val_roc_auc_list'],
            label='AUROC',
            title=f'Train and Validation AUROC, Seed Index: {seed}',
            best_epoch=extra_info['best_epoch'],
            exp_figures_dir=exp_figures_dir,
            seed=seed
        )

    if problem_type == 'multiclass':
        plot_train_val_curves(
            train_values=extra_info['train_cohen_kappa_list'],
            val_values=extra_info['val_cohen_kappa_list'],
            label='Cohen Kappa Score',
            title=f'Train and Validation Cohen Kappa Score, Seed Index: {seed}',
            best_epoch=extra_info['best_epoch'],
            exp_figures_dir=exp_figures_dir,
            seed=seed
        )

        plot_train_val_curves(
            train_values=extra_info['train_weighted_f1_list'],
            val_values=extra_info['val_weighted_f1_list'],
            label='Weighted F1',
            title=f'Train and Validation Weighted F1, Seed Index: {seed}',
            best_epoch=extra_info['best_epoch'],
            exp_figures_dir=exp_figures_dir,
            seed=seed
        )

def plot_train_val_curves(
    train_values,
    val_values,
    label,
    title,
    best_epoch,
    exp_figures_dir,
    seed
):
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label=f'Train')
    plt.plot(val_values, label=f'Validation')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{exp_figures_dir}/seed_{seed}_{label}_curve.png')
    plt.close()

def calculate_performance_metrics(problem_type, truths, probs, preds):
    performance_metrics = {}
    if problem_type == 'binary':
        balanced_acc, pr_auc, roc_auc, cm = calculate_binary_performance_metrics(
            truths=truths,
            probs=probs,
            preds=preds
        )

        performance_metrics['pr_auc'] = pr_auc
        performance_metrics['roc_auc'] = roc_auc

    elif problem_type == 'multiclass':
        balanced_acc, cohen_kappa, weighted_f1, cm = calculate_multiclass_performance_metrics(
            truths=truths,
            preds=preds
        )

        performance_metrics['cohen_kappa'] = cohen_kappa
        performance_metrics['weighted_f1'] = weighted_f1
    else:
        raise ValueError(f"Unknown classification type: {problem_type}")

    performance_metrics['balanced_accuracy'] = balanced_acc
    performance_metrics['confusion_matrix'] = cm

    return performance_metrics

def calculate_mean_and_std_performance_metrics(performance_metrics_per_seed):
    """
    Calculates the mean and std of the performance metrics across all seeds.

    Parameters
    ----------
    performance_metrics_per_seed : dict
        A dictionary where keys are random seed integers and values are dictionaries
        of performance metrics. Each inner dictionary contains metric names as keys
        (e.g., 'balanced_accuracy', 'pr_auc') and their corresponding values.
    tuple
        A tuple containing two dictionaries:
        - mean_performance_metrics: A dictionary with the mean of each metric across seeds.
        - std_performance_metrics: A dictionary with the standard deviation of each metric across seeds.
    """

    metrics = list(performance_metrics_per_seed.values())[0].keys()

    mean_performance_metrics = {}
    std_performance_metrics = {}
    for metric in metrics:
        metric_values = [performance_metrics_per_seed[seed][metric] for seed in performance_metrics_per_seed]
        mean_performance_metrics[metric] = np.mean(metric_values)
        std_performance_metrics[metric] = np.std(metric_values)

    return mean_performance_metrics, std_performance_metrics

def update_dataset_specific_params_in_exp_config(
    exp_config,
    embeddings_dir,
    processed_data_dir,
    problem_type,
    n_classes,
    monitor_metric,
    dataset_name,
    exp_name,
    exp_type
    ):

    if exp_type == 'embedding':
        exp_config['embeddings_dir'] = embeddings_dir
    elif exp_type == 'cbramod' or exp_type == 'eeg_conformer' or exp_type=='lora':
        exp_config['processed_data_dir'] = processed_data_dir
    else:
        raise ValueError()
    params = exp_config['modeling_approach_config']['params']
    early_stopping = params['early_stopping_params']

    original_tmp_dir = early_stopping['tmp_dir']
    
    params.update({
        'problem_type': problem_type,
        'n_classes': n_classes,
    })

    early_stopping.update({
        'monitor_metric': monitor_metric,
        'tmp_dir': f"{early_stopping['tmp_dir']}/{dataset_name}/{exp_name}",
    })

    os.makedirs(f"{original_tmp_dir}/{dataset_name}", exist_ok=True)
    os.makedirs(f"{original_tmp_dir}/{dataset_name}/{exp_name}", exist_ok=False)
    return exp_config