import itertools
import os
from stamp.local import get_local_config
from stamp.experiments import create_experiment

local_config = get_local_config()

experiments_dir = local_config.tsfm_experiments_dir

dataset_names = ['stress', 'bciciv2a', 'shu']

for dataset_name in dataset_names:
    exp_dir = experiments_dir + f'/{dataset_name}'
    moment_model_name = 'MOMENT-1-large'
    moment_models_dir = local_config.moment_models_dir
    n_random_seeds = 3

    training_data_ratio = 1.0
    label_smoothing = 0.1
    use_gradient_clipping = True
    use_batch_norm = False
    use_instance_norm = True

    assert use_batch_norm + use_instance_norm < 2, 'use_batch_norm and use_instance_norm should not both be True.'
    input_dim = 1024

    dropout_rate = 0.3

    n_epochs = 15
    train_batch_size = 16
    test_batch_size = 16
    min_epoch = 0
    chunk_size = None  # Use None for full sequence, or an integer for chunk size

    lora_params = {
        'r': 64,
        'alpha': 32,
        'dropout': 0.05
    }

    # Hyperparameter grids
    D_values = [128]

    # GMLP parameters (set to None to disable)
    gmlp_type = 'criss_cross'
    # gmlp_type = None
    gmlp_n_layers_values = [8] if gmlp_type is not None else None
    gmlp_dff_values = [256] if gmlp_type is not None else None
    gmlp_dr_values = [dropout_rate] if gmlp_type is not None else None
    gmlp_recurrent_values = [False] if gmlp_type is not None else None
    gmlp_combination_mode_values = ['concat'] if gmlp_type == 'criss_cross' else [None]

    # Transformer parameters (set to None to disable)
    # tr_type = 'criss_cross'
    tr_type = None
    tr_heads_values = [8] if tr_type is not None else None
    tr_n_layers_values = [2,4,8] if tr_type is not None else None
    tr_dff_values = [256,512] if tr_type is not None else None
    tr_dr_values = [dropout_rate] if tr_type is not None else None

    # Encoder aggregation
    # encoder_aggregation = 'mean_across_tokens'  # Options: 'mean_across_tokens', 'attention_pooling', 'token_prediction_averaging'
    encoder_aggregation = 'attention_pooling'  # Options: 'mean_across_tokens, 'attention_pooling', 'token_prediction_averaging'
    # encoder_aggregation = 'token_prediction_averaging'  # Options: 'mean_across_tokens', 'attention_pooling', 'token_prediction_averaging'
    mhap_A_values = [4] if encoder_aggregation == 'attention_pooling' else None
    mhap_dr_values = [dropout_rate] if encoder_aggregation == 'attention_pooling' else None
    mhap_Q_values = [8] if encoder_aggregation == 'attention_pooling' else None
    mhap_qc_values = ['weighted_sum'] if encoder_aggregation == 'attention_pooling' else None

    initial_proj_params = {
        # 'type': 'reduced',
        # 'hidden_dim': 64,
        # 'dropout_rate': 0.1,
        
        'type': 'full',
        'dropout_rate': dropout_rate,
    }

    # initial_proj_params = None  # No initial projection

    lr_params = {
        "use_scheduler": True,
        "scheduler_type": "one_cycle",
        "initial_lr": 1e-4,
        "max_lr": 1e-3, 
    }

    optimizer_params = {
        "optimizer_name": 'adamw',
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 5e-2
    }

    early_stopping_params = {
        "name": "EarlyStopping",
        "patience": 1000, # Just take the epoch with best performance, no early stopping
        "min_delta": 1e-3,
    }

    pe_params = {
        'pe_type': 'basic',
        'use_token_positional_embeddings': True,
        'use_spatial_positional_embeddings': True,
        'use_temporal_positional_embeddings': True
    }
    # pe_params = None

    # Build grid components dictionary
    grid_components = {
        'D': D_values
    }

    if gmlp_type is not None:
        grid_components.update({
            'gmlp_n_layers': gmlp_n_layers_values,
            'gmlp_dff': gmlp_dff_values,
            'gmlp_dr': gmlp_dr_values,
            'gmlp_combination_mode': gmlp_combination_mode_values,
            'gmlp_recurrent': gmlp_recurrent_values
        })

    if tr_type is not None:
        grid_components.update({
            'tr_heads': tr_heads_values,
            'tr_n_layers': tr_n_layers_values,
            'tr_dff': tr_dff_values,
            'tr_dr': tr_dr_values
        })

    if encoder_aggregation == 'attention_pooling':
        grid_components.update({
            'mhap_A': mhap_A_values,
            'mhap_dr': mhap_dr_values,
            'mhap_Q': mhap_Q_values,
            'mhap_qc': mhap_qc_values
        })

    # final_classifier_params = {
    #     'hidden_sizes': [128, 64],
    #     'dropout_rate': 0.1,
    # }

    final_classifier_params = None

    # Generate combinations using dictionary
    param_names = list(grid_components.keys())
    param_values = list(grid_components.values())
    hyperparameter_combinations = list(itertools.product(*param_values))

    print(f"Parameters: {param_names}")
    print(f"Total number of experiments: {len(hyperparameter_combinations)}")

    for i, combo in enumerate(hyperparameter_combinations, 1):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, combo))
        print(f"Creating experiment {i}/{len(hyperparameter_combinations)}")
        print(f"Parameters: {params}")

        # Extract parameters with defaults
        D = params['D']
        gmlp_n_layers = params.get('gmlp_n_layers')
        gmlp_dff = params.get('gmlp_dff') 
        gmlp_dr = params.get('gmlp_dr')
        gmlp_combination_mode = params.get('gmlp_combination_mode')
        gmlp_recurrent = params.get('gmlp_recurrent')
        tr_heads = params.get('tr_heads')
        tr_n_layers = params.get('tr_n_layers')
        tr_dff = params.get('tr_dff')
        tr_dr = params.get('tr_dr')
        mhap_A = params.get('mhap_A')
        mhap_dr = params.get('mhap_dr')
        mhap_Q = params.get('mhap_Q')
        mhap_qc = params.get('mhap_qc')

        # Create gated_mlp_params for this combination
        if gmlp_type == 'old':
            gated_mlp_params = {
                'type': gmlp_type,
                'n_layers': gmlp_n_layers,
                'dim_feedforward': gmlp_dff,
                'use_spatial_proj': True,
                'use_residual': True,
                'use_residual_gate': True,
                'dropout_rate': gmlp_dr,
                'use_t5_style': True
            }

        elif gmlp_type == 'basic':
            gated_mlp_params = {
                'type': gmlp_type,
                'n_layers': gmlp_n_layers,
                'dim_feedforward': gmlp_dff,
                'dropout_rate': gmlp_dr,
                'recurrent': gmlp_recurrent
            }

        elif gmlp_type == 'criss_cross':
            gated_mlp_params = {
                'type': gmlp_type,
                'n_layers': gmlp_n_layers,
                'dim_feedforward': gmlp_dff,
                'dropout_rate': gmlp_dr,
                'combination_mode': gmlp_combination_mode,
                'recurrent': gmlp_recurrent
            }
        else:
            gated_mlp_params = None

        if tr_type == 'basic':
            transformer_params = {
                'type': tr_type,
                'n_heads': tr_heads,
                'n_layers': tr_n_layers,
                'dim_feedforward': tr_dff,
                'dropout_rate': tr_dr,
                'activation': 'relu',
                'norm_first': False,
                'use_final_norm': True
            }
        elif tr_type == 'criss_cross':
            transformer_params = {
                'type': tr_type,
                'n_heads': tr_heads,
                'n_layers': tr_n_layers,
                'dim_feedforward': tr_dff,
                'dropout_rate': tr_dr,
                'activation': 'relu',
                'norm_first': False,
                'use_final_norm': True
            }
        else:
            transformer_params = None

        # Create mhap_params for this combination
        if encoder_aggregation == 'attention_pooling':
            mhap_params = {
                'A': mhap_A,
                'dropout_rate': mhap_dr,
                'use_residual': True,
                'n_queries_per_head': mhap_Q,
                'query_combination': mhap_qc
            }
        else:
            mhap_params = None

        exp_config = {
            'moment_model_name': moment_model_name,
            'moment_models_dir': local_config.moment_models_dir,
            'n_random_seeds': n_random_seeds,
            'training_data_ratio': training_data_ratio
        }

        modeling_approach_config = {
            'modeling_approach_name': 'MOMENTLoraAdapterModelingApproach',
            'params': {
                'moment_model_name': moment_model_name,
                'moment_models_dir': moment_models_dir,
                'lora_params': lora_params,
                'chunk_size': chunk_size,
                'label_smoothing': label_smoothing,
                'use_batch_norm': use_batch_norm,
                'use_instance_norm': use_instance_norm,
                'use_gradient_clipping': use_gradient_clipping,
                'input_dim': input_dim,
                'D': D,
                'initial_proj_params': initial_proj_params,
                'pe_params': pe_params,
                'transformer_params': transformer_params,
                'gated_mlp_params': gated_mlp_params,
                'encoder_aggregation': encoder_aggregation,
                'mhap_params': mhap_params,
                'final_classifier_params': final_classifier_params,
                'n_epochs': n_epochs,
                'train_batch_size': train_batch_size,
                'test_batch_size': test_batch_size,
                'min_epoch': min_epoch,
                "use_tqdm": False,
                "store_attention_weights": False,
                "debug_size": None,
                "lr_params": lr_params,
                "optimizer_params": optimizer_params,
                "early_stopping_params": early_stopping_params,
                'checkpointing_params': None,
            }
        }

        # Construct exp_name based on config
        exp_name = f'{moment_model_name}_LoRA-r{lora_params["r"]}-a{lora_params["alpha"]}-dr{lora_params["dropout"]}_nrs{exp_config["n_random_seeds"]}_ne{n_epochs}_D{D}_'

        # Handle initial projection
        if initial_proj_params is not None:
            exp_name += 'ip-'
            exp_name += f'{initial_proj_params["type"]}-'
            exp_name += f'dr{initial_proj_params["dropout_rate"]}'
            if initial_proj_params["type"] == 'reduced':
                exp_name += f'-H{initial_proj_params["hidden_dim"]}_'
            else:
                exp_name += '_'

        # Handle positional embeddings
        if pe_params is not None:
            exp_name += 'pe-'
            exp_name += f'{pe_params["pe_type"]}'

            if pe_params['use_token_positional_embeddings']:
                exp_name += 'N'

            if pe_params['use_spatial_positional_embeddings']:
                exp_name += 'S'

            if pe_params['use_temporal_positional_embeddings']:
                exp_name += 'T'

            exp_name += '_'

        # Gated MLP
        if gated_mlp_params is not None:
            exp_name += 'gmlp-'
            exp_name += f't{gated_mlp_params["type"]}-'
            exp_name += f'nl{gated_mlp_params["n_layers"]}-'
            exp_name += f'dff{gated_mlp_params["dim_feedforward"]}-'
            exp_name += f'dr{gated_mlp_params["dropout_rate"]}_'
            if gated_mlp_params["type"] == 'criss_cross':
                exp_name += f'cm{gated_mlp_params["combination_mode"]}_'

            if gated_mlp_params['recurrent']:
                exp_name += 'rec_'
            else:
                exp_name += 'nonrec_'

        # Transformer
        if transformer_params is not None:
            exp_name += 'tf-'
            exp_name += f't{transformer_params["type"]}-'
            exp_name += f'A{transformer_params["n_heads"]}-'
            exp_name += f'nl{transformer_params["n_layers"]}-'
            exp_name += f'dff{transformer_params["dim_feedforward"]}-'
            exp_name += f'dr{transformer_params["dropout_rate"]}_'

        # MHAP
        if mhap_params is not None:
            exp_name += 'mhap-'
            exp_name += f'A{mhap_params["A"]}-'
            exp_name += f'dr{mhap_params["dropout_rate"]}-'
            exp_name += f'Q{mhap_params["n_queries_per_head"]}-'
            exp_name += f'qc{mhap_params["query_combination"]}_'
        else:
            exp_name += f'ea-{encoder_aggregation}_'

        if final_classifier_params is not None:
            exp_name += 'fc-'
            exp_name += 'h'
            for h in final_classifier_params['hidden_sizes']:
                exp_name += f'{h}'
            exp_name += f'-dr{final_classifier_params["dropout_rate"]}_'

        # Label Smoothing
        if label_smoothing is not None:
            exp_name += f'ls{label_smoothing}'

        # Gradient Clipping
        exp_name += f"gc{'T' if use_gradient_clipping else 'F'}_"
        
        if use_batch_norm:
            exp_name += f"bnorm_"
        
        if use_instance_norm:
            exp_name += f"inorm_"
        
        exp_name += f'tdr{training_data_ratio}'

        if os.path.exists(experiments_dir + f'/{dataset_name}/{exp_name}'):
            continue

        # Create a copy of early_stopping_params for each experiment
        early_stopping_params_copy = early_stopping_params.copy()
        early_stopping_params_copy['tmp_dir'] = local_config.tmp_dir

        # Update the modeling approach config with the copied early stopping params
        modeling_approach_config['params']['early_stopping_params'] = early_stopping_params_copy
        modeling_approach_config['params']['checkpointing_params'] = {
            'checkpoint_dir': None
        }

        exp_config['modeling_approach_config'] = modeling_approach_config
        final_exp_config = {'exp_name': exp_name}
        final_exp_config.update(exp_config)

        # Create the experiment
        create_experiment(exp_dir=exp_dir, exp_config=final_exp_config)

    print(f"Successfully created {len(hyperparameter_combinations)} experiments!")