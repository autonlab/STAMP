from stamp.local import get_local_config
from stamp.experiments import create_experiment

local_config = get_local_config()

experiments_dir = '/path/to/cbramod/experiments'

dataset_name = 'tuev'
tdr = 0.3  # Training Data Ratio
exp_dir = experiments_dir + f'/{dataset_name}'

modeling_approach_config = {
    'modeling_approach_name': 'CBraModModelingApproach',
    'params': {
        'dataset_name': dataset_name,
        'processed_data_dir': local_config.processed_data_dirs[dataset_name],
        'n_epochs': 50,
        'use_gradient_clipping': True,
        'train_batch_size': 64,
        'test_batch_size': 64,
        'min_epoch': 0,
        'label_smoothing': 0.1,
        'lr_params': {
            'initial_lr': 1e-4,
            'use_scheduler': True,
        },
        'optimizer_params': {
            'weight_decay': 5e-2,
        },
        'early_stopping_params': {
            'tmp_dir': '/path/to/tmp_experiments'
        },
        'model_params': { # Pass directly to model
            'classifier': 'all_patch_reps',
            'frozen': False,
            'use_pretrained_weights': True,
            'dropout': 0.1,
            'multi_lr': True,
            'foundation_dir': '/path/to/cbramod/pretrained_weights/pretrained_weights.pth'
        }
    }
}

if modeling_approach_config['params']['model_params']['multi_lr']:
    mlr_label = 'mlr-T'
else:
    mlr_label = 'mlr-F'

exp_config = {
    'exp_name': f'CBraMod_experiment_{mlr_label}_tdr{tdr}',
    'n_random_seeds': 5,
    'training_data_ratio': tdr,
    'modeling_approach_config': modeling_approach_config,
}

# Create the experiment
create_experiment(exp_dir=exp_dir, exp_config=exp_config)

print(f"Successfully created experiment!")