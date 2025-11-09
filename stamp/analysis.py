import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from stamp.experiments import load_experiment_config
from stamp.local import get_local_config

local_config = get_local_config()

proper_dataset_name_map = {
        'shu': 'SHU-MI (2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic (2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a (4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI (4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016 (2-Class, 7,143 Samples)',
        'seedv': 'SEED-V (5-Class, 117,744 Samples)',
        'tuev': 'TUEV (6-Class, 113,353 Samples)',
        'faced': 'FACED (9-Class, 10,332 Samples)',
    }

cbramod_reported_performance_per_dataset = {
    'shu': {
        'balanced_accuracy': (0.6370, 0.0151),
        'pr_auc': (0.7139, 0.0088),
        'roc_auc': (0.6988, 0.0068)
    },
    'stress': {
        'balanced_accuracy': (0.7256, 0.0132),
        'pr_auc': (0.6267, 0.0099),
        'roc_auc': (0.7905, 0.0073)
    },
    'mumtaz': {
        'balanced_accuracy': (0.9560,0.0056),
        'pr_auc': (0.9923,0.0032),
        'roc_auc': (0.9921,0.0025)
    },
    'speech': {
        'balanced_accuracy': (0.5373, 0.0108),
        'cohen_kappa': (0.4216, 0.0163),
        'weighted_f1': (0.5383, 0.0096)
    },
    'physio': {
        'balanced_accuracy': (0.6417, 0.0091),
        'cohen_kappa': (0.5222, 0.0169),
        'weighted_f1': (0.6427, 0.0100)
    },
    'seedv': {
        'balanced_accuracy': (0.4091, 0.0097),
        'cohen_kappa': (0.2569, 0.0143),
        'weighted_f1': (0.4101, 0.0108)
    },
    'bciciv2a': {
        'balanced_accuracy': (0.5138, 0.0066),
        'cohen_kappa': (0.3518, 0.0094),
        'weighted_f1': (0.4984, 0.0085)
    },
    'isruc': {
        'balanced_accuracy': (0.7865, 0.0110),
        'cohen_kappa': (0.7442, 0.0152),
        'weighted_f1': (0.8011, 0.0099)
    },
    'tuev': {
        'balanced_accuracy': (0.6659, 0.0124),
        'cohen_kappa': (0.6744, 0.0121),
        'weighted_f1': (0.8331, 0.0071)
    },
    'faced': {
        'balanced_accuracy': (0.5509, 0.0089),
        'cohen_kappa': (0.5041, 0.0122),
        'weighted_f1': (0.5618, 0.0093)
    },

}

labram_reported_performance_per_dataset = {
    'shu': {
        'balanced_accuracy': (0.6166, 0.0192),
        'pr_auc': (0.6761, 0.0083),
        'roc_auc': (0.6604, 0.0091),
    },

    'stress': {
        'balanced_accuracy': (0.6909, 0.0125),
        'pr_auc': (0.5999, 0.0155),
        'roc_auc': (0.7721, 0.0093)
    },

    'physio': {
        'balanced_accuracy': (0.6173, 0.0122),
        'cohen_kappa': (0.4912, 0.0192),
        'weighted_f1': (0.6177, 0.0141)
    },

    'faced': {
        'balanced_accuracy': (0.5273, 0.0107),
        'cohen_kappa': (0.4698, 0.0188),
        'weighted_f1': (0.5288, 0.0102)
    },

    'seedv': {
        'balanced_accuracy': (0.3976, 0.0138),
        'cohen_kappa': (0.2386, 0.0209),
        'weighted_f1': (0.3974, 0.0111)
    },
    'bciciv2a': {
        'balanced_accuracy': (0.4869, 0.0085),
        'cohen_kappa': (0.3159, 0.0154),
        'weighted_f1': (0.4758, 0.0103)
    },
    'isruc': {
        'balanced_accuracy': (0.7633, 0.0102),
        'cohen_kappa': (0.7231, 0.0182),
        'weighted_f1': (0.7810, 0.0133)
    },
    'tuev': {
        'balanced_accuracy': (0.6409, 0.0065),
        'cohen_kappa': (0.6637, 0.0093),
        'weighted_f1': (0.8312, 0.0052)
    },
    'faced': {
        'balanced_accuracy': (0.5273, 0.0107),
        'cohen_kappa': (0.4698, 0.0188),
        'weighted_f1': (0.5288, 0.0102)
    },
    'mumtaz': {
        'balanced_accuracy': (0.9409, 0.0079),
        'pr_auc': (0.9798, 0.0093),
        'roc_auc': (0.9782, 0.0057)
    }
}

biot_reported_performance_per_dataset = {
    'shu': {
        'balanced_accuracy': (0.6179, 0.0183),
        'pr_auc': (0.6770, 0.0119),
        'roc_auc': (0.6609, 0.0127),
    },
    'physio': {
        'balanced_accuracy': (0.6153, 0.0154),
        'cohen_kappa': (0.4875, 0.0272),
        'weighted_f1': (0.6158, 0.0197)
    },
    'isruc': {
        'balanced_accuracy': (0.7527, 0.0121),
        'cohen_kappa': (0.7192, 0.0231),
        'weighted_f1': (0.7790, 0.0146)
    },
    'stress': {
        'balanced_accuracy': (0.6875, 0.0186),
        'pr_auc': (0.6004, 0.0195),
        'roc_auc': (0.7536, 0.0144)
    },
    'seedv': {
        'balanced_accuracy': (0.3837, 0.0187),
        'cohen_kappa': (0.2261, 0.0262),
        'weighted_f1': (0.3856, 0.0203)
    },
    'tuev': {
        'balanced_accuracy': (0.5281, 0.0225),
        'cohen_kappa': (0.5273, 0.0249),
        'weighted_f1': (0.7492, 0.0082)
    },
    'bciciv2a': {
        'balanced_accuracy': (0.4748, 0.0093),
        'cohen_kappa': (0.2997, 0.0139),
        'weighted_f1': (0.4607, 0.0125)
    },
    'faced': {
        'balanced_accuracy': (0.5118, 0.0118),
        'cohen_kappa': (0.4476, 0.0254),
        'weighted_f1': (0.5136, 0.0112)
    },
    'mumtaz': {
        'balanced_accuracy': (0.9358, 0.0052),
        'pr_auc': (0.9736, 0.0034),
        'roc_auc': (0.9758, 0.0042)
    }
}

st_transformer_reported_performance_per_dataset = {
    'shu': {
        'balanced_accuracy': (0.5992, 0.0206),
        'pr_auc': (0.6394, 0.0122),
        'roc_auc': (0.6431, 0.0111),
    },
    'physio': {
        'balanced_accuracy': (0.6035, 0.0081),
        'cohen_kappa': (0.4712, 0.0199),
        'weighted_f1': (0.6053, 0.0075)
    },
    'isruc': {
        'balanced_accuracy': (0.7381, 0.0205),
        'cohen_kappa': (0.7013, 0.0352),
        'weighted_f1': (0.7681, 0.0175)
    },
    'stress': {
        'balanced_accuracy': (0.6631, 0.0173),
        'pr_auc': (0.5672, 0.0259),
        'roc_auc': (0.7132, 0.0174)
    },
    'seedv': {
        'balanced_accuracy': (0.3052, 0.0072),
        'cohen_kappa': (0.1083, 0.0121),
        'weighted_f1': (0.2833, 0.0105)
    },
    'tuev': {
        'balanced_accuracy': (0.3984, 0.0228),
        'cohen_kappa': (0.3765, 0.0306),
        'weighted_f1': (0.6823, 0.0190)
    },
    'bciciv2a': {
        'balanced_accuracy': (0.4575, 0.0145),
        'cohen_kappa': (0.2733, 0.0198),
        'weighted_f1': (0.4471, 0.0142)
    },
    'faced': {
        'balanced_accuracy': (0.4810, 0.0079),
        'cohen_kappa': (0.4137, 0.0133),
        'weighted_f1': (0.4795, 0.0096)
    },
    'mumtaz': {
        'balanced_accuracy': (0.9135, 0.0103),
        'pr_auc': (0.9578, 0.0086),
        'roc_auc': (0.9594, 0.0059)
    },

}

eeg_conformer_reported_performance_per_dataset = {
    'shu': {
        'balanced_accuracy': (0.5900, 0.0107),
        'pr_auc': (0.6370, 0.0093),
        'roc_auc': (0.6351, 0.0101),
    },
    'physio': {
        'balanced_accuracy': (0.6049, 0.0104),
        'cohen_kappa': (0.4736, 0.0171),
        'weighted_f1': (0.6062, 0.0095)
    },
    'isruc': {
        'balanced_accuracy': (0.7400, 0.0133),
        'cohen_kappa': (0.7143, 0.0162),
        'weighted_f1': (0.7634, 0.0151)
    },
    'stress': {
        'balanced_accuracy': (0.6805, 0.0123),
        'pr_auc': (0.5829, 0.0134),
        'roc_auc': (0.7424, 0.0128)
    },
    'seedv': {
        'balanced_accuracy': (0.3537, 0.0112),
        'cohen_kappa': (0.1772, 0.0174),
        'weighted_f1': (0.3487, 0.0136)
    },
    'tuev': {
        'balanced_accuracy': (0.4074, 0.0164),
        'cohen_kappa': (0.3967, 0.0195),
        'weighted_f1': (0.6983, 0.0152)
    },
    'bciciv2a': {
        'balanced_accuracy': (0.4696, 0.0106),
        'cohen_kappa': (0.2924, 0.0141),
        'weighted_f1': (0.4533, 0.0128)
    },
    'faced': {
        'balanced_accuracy': (0.4559, 0.0125),
        'cohen_kappa': (0.3858, 0.0186),
        'weighted_f1': (0.4514, 0.0107)
    },
    'mumtaz': {
        'balanced_accuracy': (0.9308, 0.0117),
        'pr_auc': (0.9684, 0.0105),
        'roc_auc': (0.9702, 0.0101)
    }
}

def get_val_performance(
    experiments_dir,
    exp,
    problem_type
):
    # Load extra_info_per_seed.pkl
    extra_info_per_seed = pd.read_pickle(os.path.join(experiments_dir, exp, 'results', 'extra_info_per_seed.pkl'))
    
    metrics = {}
    
    # Get metrics based on problem type
    if problem_type == 'binary':
        metrics = ['balanced_accuracy', 'pr_auc', 'roc_auc']
    elif problem_type == 'multiclass':
        metrics = ['balanced_accuracy', 'cohen_kappa', 'weighted_f1']
    elif problem_type == 'regression':
        pass
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

    # Gather performance metrics (at best epoch) across seeds
    performance_dict = {}
    for seed, extra_info in extra_info_per_seed.items():
        print(f'Seed: {seed}')
        best_epoch = extra_info['best_epoch']

        for metric in metrics:
            if metric not in performance_dict:
                performance_dict[metric] = []

            metric_label = metric
            if metric == 'balanced_accuracy':
                metric_label = 'balanced_acc'
            metric_val = extra_info[f'val_{metric_label}_list'][best_epoch]

            performance_dict[metric].append(metric_val)

    mean_metrics = {metric: np.mean(vals) for metric, vals in performance_dict.items()}
    std_metrics = {metric: np.std(vals) for metric, vals in performance_dict.items()}
    return mean_metrics, std_metrics

def get_test_performance(
    mean_path,
    std_path
):
    mean_metrics = pd.read_pickle(mean_path)
    std_metrics = pd.read_pickle(std_path)
    return mean_metrics, std_metrics

def add_reported_data(available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
                          show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported):

        if show_reported_cbramod:
            available_experiments.append('CBramod')
            mean_data['CBramod'] = {k: v[0] for k, v in cbramod_reported_performance_per_dataset[dataset_name].items()}
            std_data['CBramod'] = {k: v[1] for k, v in cbramod_reported_performance_per_dataset[dataset_name].items()}
            total_params_per_exp['CBramod'] = 4000000
            is_reported['CBramod'] = True
            labels = ['CBramod', *labels]
            label_lookup['CBramod'] = 'CBramod'
            
        if show_reported_labram:
            available_experiments.append('LaBraM')
            mean_data['LaBraM'] = {k: v[0] for k, v in labram_reported_performance_per_dataset[dataset_name].items()}
            std_data['LaBraM'] = {k: v[1] for k, v in labram_reported_performance_per_dataset[dataset_name].items()}
            total_params_per_exp['LaBraM'] = 5800000
            is_reported['LaBraM'] = True
            labels = ['LaBraM', *labels]
            label_lookup['LaBraM'] = 'LaBraM'
            
        if show_biot_reported:
            available_experiments.append('BIOT')
            mean_data['BIOT'] = {k: v[0] for k, v in biot_reported_performance_per_dataset[dataset_name].items()}
            std_data['BIOT'] = {k: v[1] for k, v in biot_reported_performance_per_dataset[dataset_name].items()}
            total_params_per_exp['BIOT'] = 3200000
            is_reported['BIOT'] = True
            labels = ['BIOT', *labels]
            label_lookup['BIOT'] = 'BIOT'
            
        if show_st_transformer_reported:
            available_experiments.append('ST-Transformer')
            mean_data['ST-Transformer'] = {k: v[0] for k, v in st_transformer_reported_performance_per_dataset[dataset_name].items()}
            std_data['ST-Transformer'] = {k: v[1] for k, v in st_transformer_reported_performance_per_dataset[dataset_name].items()}
            total_params_per_exp['ST-Transformer'] = 3500000
            is_reported['ST-Transformer'] = True
            labels = ['ST-Transformer', *labels]
            label_lookup['ST-Transformer'] = 'ST-Transformer'
            
        if show_eeg_conformer_reported:
            available_experiments.append('EEG Conformer')
            mean_data['EEG Conformer'] = {k: v[0] for k, v in eeg_conformer_reported_performance_per_dataset[dataset_name].items()}
            std_data['EEG Conformer'] = {k: v[1] for k, v in eeg_conformer_reported_performance_per_dataset[dataset_name].items()}
            total_params_per_exp['EEG Conformer'] = 550000
            is_reported['EEG Conformer'] = True
            labels = ['EEG Conformer', *labels]
            label_lookup['EEG Conformer'] = 'EEG Conformer'

        return available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup

def plot_top_25_percent_experiments_hbar(dataset_name, experiments_dir, label_per_experiment, split, cbramod_experiments_dir=None, metrics_to_plot=None, figsize=(12, 8),  
                                         show_reported_cbramod=True, show_reported_labram=True, show_biot_reported=True, show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Plots mean performance metrics with standard deviation error bars for the top 25%
    of non-reported experiments plus all reported experiments using horizontal bar charts.
    Top experiments are selected by highest roc_auc when available, otherwise cohen_kappa.
    """

    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}  # Track which experiments are "reported"

    available_experiments = []
    experimental_experiments = []  # Track experimental experiments separately

    # Add reported experiments first
    available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported)

    # Load all experimental results first to find the top 25%
    experimental_data = {}
    experimental_scores = {}
    
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue

        total_params = exp_config['total_parameters']

        if split == 'val':
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        experimental_data[exp] = {
            'mean': mean_metrics,
            'std': std_metrics,
            'params': total_params
        }
        
        # Calculate score for ranking
        if 'roc_auc' in mean_metrics:
            score = mean_metrics['roc_auc']
            metric_used = 'roc_auc'
        elif 'cohen_kappa' in mean_metrics:
            score = mean_metrics['cohen_kappa']
            metric_used = 'cohen_kappa'
        else:
            print(f"Warning: Neither roc_auc nor cohen_kappa found for {exp}")
            continue
            
        experimental_scores[exp] = {'score': score, 'metric': metric_used}
        experimental_experiments.append(exp)

    if not experimental_data:
        print("Warning: No experimental results found. Showing only reported results.")
        top_experiments = []
    else:
        # Sort experiments by score (descending) and select top 25%
        sorted_experiments = sorted(experimental_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Calculate how many experiments to include (at least 1, up to all if less than 4 total)
        num_experiments = len(sorted_experiments)
        num_top_25_percent = max(1, int(np.ceil(num_experiments * 0.25)))
        
        top_experiments = [exp for exp, _ in sorted_experiments[:num_top_25_percent]]
        
        print(f"\nTotal experimental experiments: {num_experiments}")
        print(f"Top 25% count: {num_top_25_percent}")
        print(f"Selected top experiments:")
        
        for i, (exp, data) in enumerate(sorted_experiments[:num_top_25_percent]):
            print(f"  {i+1}. {exp}: {data['metric']} = {data['score']:.4f}")
        
        # Add the top experiments to our plotting data
        for exp in top_experiments:
            available_experiments.append(exp)
            mean_data[exp] = experimental_data[exp]['mean']
            std_data[exp] = experimental_data[exp]['std']
            total_params_per_exp[exp] = experimental_data[exp]['params']
            is_reported[exp] = False

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")

    # Determine metrics to plot
    all_metrics = list(next(iter(mean_data.values())).keys())
    if metrics_to_plot is None:
        metrics_to_plot = all_metrics

    # Filter out confusion_matrix from plotting metrics
    plot_metrics = [metric for metric in metrics_to_plot if metric != 'confusion_matrix']
    
    # Limit to 3 metrics for the subplot layout
    if len(plot_metrics) > 3:
        plot_metrics = plot_metrics[:3]
        print(f"Warning: Only plotting first 3 metrics: {plot_metrics}")
    
    # Create total_params_list with proper structure
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    # Define colors and styles for different types
    experimental_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    reported_colors = ['blue', 'red', 'orange', 'purple', 'green']
    
    # Create figure with subplots: 2 on top row, 1 on bottom row
    fig = plt.figure(figsize=figsize)
    
    # Create subplot layout
    if len(plot_metrics) == 1:
        # If only one metric, use the full figure
        axes = [plt.subplot(1, 1, 1)]
    elif len(plot_metrics) == 2:
        # If two metrics, place them side by side on top row
        axes = [plt.subplot(2, 2, 1), plt.subplot(2, 2, 2)]
    else:
        # Three metrics: 2 on top, 1 on bottom (spanning both columns)
        axes = [
            plt.subplot(2, 2, 1),  # Top left
            plt.subplot(2, 2, 2),  # Top right
            plt.subplot(2, 1, 2)   # Bottom (spans both columns)
        ]
    
    # Plot each metric in its respective subplot
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        plt.sca(ax)  # Set current axis
        
        means = [mean_data[exp][metric] for exp in available_experiments]
        stds = [std_data[exp][metric] for exp in available_experiments]

        y = np.arange(len(available_experiments))
        
        # Create bars with different styling
        bars = []
        colors_used = []
        reported_idx = 0
        experimental_idx = 0
        
        for i, exp in enumerate(available_experiments):
            if is_reported[exp]:
                # Reported bars: colored with hatching pattern
                color = reported_colors[reported_idx % len(reported_colors)]
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.3, hatch='///', edgecolor='black', linewidth=1)
                reported_idx += 1
            else:
                # Experimental bars: colored, solid with gradient alpha based on ranking
                color = experimental_colors[experimental_idx % len(experimental_colors)]
                
                # Add gradient effect: best experiments get higher alpha
                exp_rank = top_experiments.index(exp) if exp in top_experiments else 0
                alpha = 1.0 - (exp_rank * 0.15)  # Decrease alpha slightly for lower ranks
                alpha = max(alpha, 0.6)  # Minimum alpha of 0.6
                
                # Highlight top performer with thicker border
                linewidth = 2 if exp_rank == 0 else 1
                
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=alpha, edgecolor='black', linewidth=linewidth)
                experimental_idx += 1
            
            bars.extend(bar)
            colors_used.append(color)

        # Add parameter counts as text on each bar
        upper_ci = [m + s for m, s in zip(means, stds)]
        for i, (bar, params) in enumerate(zip(bars, total_params_list)):
            # Format parameter count
            if params >= 1000000:
                param_text = f"{params/1000000:.1f}M"
            elif params >= 1000:
                param_text = f"{params/1000:.1f}K"
            else:
                param_text = str(params)

            # Position text
            bar_width = bar.get_width()
            text_x = bar_width + (max(upper_ci) * -0.18)

            ax.text(text_x, bar.get_y() + bar.get_height()/2,
                    param_text,
                    ha='left', va='center',
                    fontweight='bold', fontsize=9)
        
        ytick_labels = [label_lookup.get(exp, exp) for exp in available_experiments]
        ax.set_yticks(y)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel(metric)
        
        # Update title to indicate top 25% selection
        title = f"{metric} for {dataset_name} - {split}"
        if top_experiments:
            title += f"\n(Top 25% Experimental: {len(top_experiments)} experiments)"
        ax.set_title(title)
        ax.invert_yaxis()
    
    # Create custom legend for the entire figure
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.9, 
                         edgecolor='black', linewidth=1, label=f'Top 25% Experimental Results (n={len(top_experiments) if top_experiments else 0})'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.6, 
                         hatch='///', edgecolor='black', linewidth=1, label='Reported Results')
    ]
    
    # Place legend at the bottom of the figure
    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend
    plt.show()

def plot_best_experiment_metrics_hbar(dataset_name, experiments_dir, label_per_experiment, split, cbramod_experiments_dir=None, metrics_to_plot=None, figsize=(10, 6),  
                                     show_reported_cbramod=True, show_reported_labram=True, show_biot_reported=True, show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Plots mean performance metrics with standard deviation error bars for the best performing
    non-reported experiment plus all reported experiments using horizontal bar charts.
    Best experiment is selected by highest roc_auc when available, otherwise cohen_kappa.
    """

    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}  # Track which experiments are "reported"

    available_experiments = []
    experimental_experiments = []  # Track experimental experiments separately

    # Add reported experiments first
    available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported)
    
    # Load all experimental results first to find the best one
    experimental_data = {}
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue

        total_params = exp_config['total_parameters']

        if split == 'val':
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        experimental_data[exp] = {
            'mean': mean_metrics,
            'std': std_metrics,
            'params': total_params
        }
        experimental_experiments.append(exp)

    if not experimental_data:
        print("Warning: No experimental results found. Showing only reported results.")
        best_exp = None
    else:
        # Find the best experimental experiment
        best_exp = None
        best_score = -float('inf')
        
        for exp, data in experimental_data.items():
            # Try roc_auc first, then cohen_kappa
            if 'roc_auc' in data['mean']:
                score = data['mean']['roc_auc']
                print(f"Experiment {exp}: roc_auc = {score:.4f}")
            elif 'cohen_kappa' in data['mean']:
                score = data['mean']['cohen_kappa']
                print(f"Experiment {exp}: cohen_kappa = {score:.4f}")
            else:
                print(f"Warning: Neither roc_auc nor cohen_kappa found for {exp}")
                continue
            
            if score > best_score:
                best_score = score
                best_exp = exp
        
        if best_exp:
            print(f"\nBest experimental experiment: {best_exp} with score: {best_score:.4f}")
            
            # Add the best experiment to our plotting data
            available_experiments.append(best_exp)
            mean_data[best_exp] = experimental_data[best_exp]['mean']
            std_data[best_exp] = experimental_data[best_exp]['std']
            total_params_per_exp[best_exp] = experimental_data[best_exp]['params']
            is_reported[best_exp] = False

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")

    # Determine metrics to plot
    all_metrics = list(next(iter(mean_data.values())).keys())
    if metrics_to_plot is None:
        metrics_to_plot = all_metrics

    # Filter out confusion_matrix from plotting metrics
    plot_metrics = [metric for metric in metrics_to_plot if metric != 'confusion_matrix']
    
    # Limit to 3 metrics for the subplot layout
    if len(plot_metrics) > 3:
        plot_metrics = plot_metrics[:3]
        print(f"Warning: Only plotting first 3 metrics: {plot_metrics}")
    
    # Create total_params_list with proper structure
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    # Define colors and styles for different types
    experimental_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    reported_colors = ['blue', 'red', 'orange', 'purple', 'green']
    
    # Create figure with subplots: 2 on top row, 1 on bottom row
    fig = plt.figure(figsize=figsize)
    
    # Create subplot layout
    if len(plot_metrics) == 1:
        # If only one metric, use the full figure
        axes = [plt.subplot(1, 1, 1)]
    elif len(plot_metrics) == 2:
        # If two metrics, place them side by side on top row
        axes = [plt.subplot(2, 2, 1), plt.subplot(2, 2, 2)]
    else:
        # Three metrics: 2 on top, 1 on bottom (spanning both columns)
        axes = [
            plt.subplot(2, 2, 1),  # Top left
            plt.subplot(2, 2, 2),  # Top right
            plt.subplot(2, 1, 2)   # Bottom (spans both columns)
        ]
    
    # Plot each metric in its respective subplot
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        plt.sca(ax)  # Set current axis
        
        means = [mean_data[exp][metric] for exp in available_experiments]
        stds = [std_data[exp][metric] for exp in available_experiments]

        y = np.arange(len(available_experiments))
        
        # Create bars with different styling
        bars = []
        colors_used = []
        reported_idx = 0
        experimental_idx = 0
        
        for i, exp in enumerate(available_experiments):
            if is_reported[exp]:
                # Reported bars: colored with hatching pattern
                color = reported_colors[reported_idx % len(reported_colors)]
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.3, hatch='///', edgecolor='black', linewidth=1)
                reported_idx += 1
            else:
                # Experimental bars: colored, solid (highlight the best one)
                color = experimental_colors[experimental_idx % len(experimental_colors)]
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.8, edgecolor='black', linewidth=2)  # Thicker border for best
                experimental_idx += 1
            
            bars.extend(bar)
            colors_used.append(color)

        # Add parameter counts as text on each bar
        upper_ci = [m + s for m, s in zip(means, stds)]
        for i, (bar, params) in enumerate(zip(bars, total_params_list)):
            # Format parameter count
            if params >= 1000000:
                param_text = f"{params/1000000:.1f}M"
            elif params >= 1000:
                param_text = f"{params/1000:.1f}K"
            else:
                param_text = str(params)

            # Position text
            bar_width = bar.get_width()
            text_x = bar_width + (max(upper_ci) * -0.18)

            ax.text(text_x, bar.get_y() + bar.get_height()/2,
                    param_text,
                    ha='left', va='center',
                    fontweight='bold', fontsize=9)
        
        ytick_labels = [label_lookup.get(exp, exp) for exp in available_experiments]
        ax.set_yticks(y)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel(metric)
        
        # Update title to indicate best experiment selection
        title = f"{metric} for {dataset_name} - {split}"
        if best_exp:
            title += f"\n(Best Experimental: {label_lookup.get(best_exp, best_exp)})"
        ax.set_title(title)
        ax.invert_yaxis()
    
    # Create custom legend for the entire figure
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, 
                         edgecolor='black', linewidth=2, label='Best Experimental Result'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.6, 
                         hatch='///', edgecolor='black', linewidth=1, label='Reported Results')
    ]
    
    # Place legend at the bottom of the figure
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    plt.show()

def plot_experiment_metrics_hbar_no_groups_distinct_single_plot(dataset_name, experiments_dir, label_per_experiment, split, cbramod_experiments_dir=None, metrics_to_plot=None, figsize=(10, 6),  
                                           show_reported_cbramod=True, show_reported_labram=True, show_biot_reported=True, show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Plots mean performance metrics with standard deviation error bars for each experiment
    using horizontal bar charts with distinct styling for reported vs experimental results.
    """

    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}  # Track which experiments are "reported"

    available_experiments = []

    # Add reported experiments first
    available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported)

    # Add experimental results
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue
        else:
            available_experiments.append(exp)
            is_reported[exp] = False  # Mark as experimental

        total_params = exp_config['total_parameters']

        if split == 'val':
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        print(f"Experiment: {exp}"
              f"\nMean Metrics: {mean_metrics}"
              f"\nStd Metrics: {std_metrics}\n")

        mean_data[exp] = mean_metrics
        std_data[exp] = std_metrics
        total_params_per_exp[exp] = total_params

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")

    # Determine metrics to plot
    all_metrics = list(next(iter(mean_data.values())).keys())
    if metrics_to_plot is None:
        metrics_to_plot = all_metrics

    # Filter out confusion_matrix from plotting metrics
    plot_metrics = [metric for metric in metrics_to_plot if metric != 'confusion_matrix']
    
    # Limit to 3 metrics for the subplot layout
    if len(plot_metrics) > 3:
        plot_metrics = plot_metrics[:3]
        print(f"Warning: Only plotting first 3 metrics: {plot_metrics}")
    
    # Create total_params_list with proper structure
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    # Define colors and styles for different types
    experimental_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    reported_colors = ['blue', 'red', 'orange', 'purple', 'green']
    
    # Create figure with subplots: 2 on top row, 1 on bottom row
    fig = plt.figure(figsize=figsize)
    
    # Create subplot layout
    if len(plot_metrics) == 1:
        # If only one metric, use the full figure
        axes = [plt.subplot(1, 1, 1)]
    elif len(plot_metrics) == 2:
        # If two metrics, place them side by side on top row
        axes = [plt.subplot(2, 2, 1), plt.subplot(2, 2, 2)]
    else:
        # Three metrics: 2 on top, 1 on bottom (spanning both columns)
        axes = [
            plt.subplot(2, 2, 1),  # Top left
            plt.subplot(2, 2, 2),  # Top right
            plt.subplot(2, 1, 2)   # Bottom (spans both columns)
        ]
    
    # Plot each metric in its respective subplot
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        plt.sca(ax)  # Set current axis
        
        means = [mean_data[exp][metric] for exp in available_experiments]
        stds = [std_data[exp][metric] for exp in available_experiments]

        y = np.arange(len(available_experiments))
        
        # Create bars with different styling
        bars = []
        colors_used = []
        reported_idx = 0
        experimental_idx = 0
        
        for i, exp in enumerate(available_experiments):
            if is_reported[exp]:
                # Reported bars: colored with hatching pattern
                color = reported_colors[reported_idx % len(reported_colors)]
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.3, hatch='///', edgecolor='black', linewidth=1)
                reported_idx += 1
            else:
                # Experimental bars: colored, solid
                color = experimental_colors[experimental_idx % len(experimental_colors)]
                bar = ax.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                experimental_idx += 1
            
            bars.extend(bar)
            colors_used.append(color)

        # Add parameter counts as text on each bar
        upper_ci = [m + s for m, s in zip(means, stds)]
        for i, (bar, params) in enumerate(zip(bars, total_params_list)):
            # Format parameter count
            if params >= 1000000:
                param_text = f"{params/1000000:.1f}M"
            elif params >= 1000:
                param_text = f"{params/1000:.1f}K"
            else:
                param_text = str(params)

            # Position text
            bar_width = bar.get_width()
            text_x = bar_width + (max(upper_ci) * -0.18)

            ax.text(text_x, bar.get_y() + bar.get_height()/2,
                    param_text,
                    ha='left', va='center',
                    fontweight='bold', fontsize=9)
        
        ytick_labels = [label_lookup[exp] for exp in available_experiments]
        ax.set_yticks(y)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} for {dataset_name} - {split}")
        ax.invert_yaxis()
    
    # Create custom legend for the entire figure
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, 
                         edgecolor='black', linewidth=0.5, label='Experimental Results'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.6, 
                         hatch='///', edgecolor='black', linewidth=1, label='Reported Results')
    ]
    
    # Place legend at the bottom of the figure
    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    plt.show()

def plot_experiment_metrics_hbar_no_groups_distinct(dataset_name, experiments_dir, label_per_experiment, split, cbramod_experiments_dir=None, metrics_to_plot=None, figsize=(10, 6),  
                                           show_reported_cbramod=True, show_reported_labram=True, show_biot_reported=True, show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Plots mean performance metrics with standard deviation error bars for each experiment
    using horizontal bar charts with distinct styling for reported vs experimental results.
    """

    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}  # Track which experiments are "reported"

    available_experiments = []

    # Add reported experiments first
    available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported)

    # Add experimental results
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue
        else:
            available_experiments.append(exp)
            is_reported[exp] = False  # Mark as experimental

        total_params = exp_config['total_parameters']

        if split == 'val':
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        print(f"Experiment: {exp}"
              f"\nMean Metrics: {mean_metrics}"
              f"\nStd Metrics: {std_metrics}\n")

        mean_data[exp] = mean_metrics
        std_data[exp] = std_metrics
        total_params_per_exp[exp] = total_params

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")

    # Determine metrics to plot
    all_metrics = list(next(iter(mean_data.values())).keys())
    if metrics_to_plot is None:
        metrics_to_plot = all_metrics

    # Create total_params_list with proper structure
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    # Define colors and styles for different types
    experimental_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    reported_colors = ['blue', 'red', 'orange', 'purple', 'green']  # Grayscale for reported
    
    # Plot each metric separately
    for metric in metrics_to_plot:
        if metric == 'confusion_matrix':
            continue
            
        means = [mean_data[exp][metric] for exp in available_experiments]
        stds = [std_data[exp][metric] for exp in available_experiments]

        y = np.arange(len(available_experiments))

        plt.figure(figsize=figsize)
        
        # Create bars with different styling
        bars = []
        colors_used = []
        reported_idx = 0
        experimental_idx = 0
        
        for i, exp in enumerate(available_experiments):
            if is_reported[exp]:
                # Reported bars: grayscale with hatching pattern
                color = reported_colors[reported_idx % len(reported_colors)]
                bar = plt.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.3, hatch='///', edgecolor='black', linewidth=1)
                reported_idx += 1
            else:
                # Experimental bars: colored, solid
                color = experimental_colors[experimental_idx % len(experimental_colors)]
                bar = plt.barh(y[i], means[i], xerr=stds[i], capsize=5, 
                             color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                experimental_idx += 1
            
            bars.extend(bar)
            colors_used.append(color)

        # Add parameter counts as text on each bar
        upper_ci = [m + s for m, s in zip(means, stds)]
        for i, (bar, params) in enumerate(zip(bars, total_params_list)):
            # Format parameter count
            if params >= 1000000:
                param_text = f"{params/1000000:.1f}M"
            elif params >= 1000:
                param_text = f"{params/1000:.1f}K"
            else:
                param_text = str(params)

            # Position text
            bar_width = bar.get_width()
            text_x = bar_width + (max(upper_ci) * -0.18)

            plt.text(text_x, bar.get_y() + bar.get_height()/2,
                    param_text,
                    ha='left', va='center',
                    fontweight='bold', fontsize=9)
        
        # Create custom legend
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, 
                             edgecolor='black', linewidth=0.5, label='Experimental Results'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.6, 
                             hatch='///', edgecolor='black', linewidth=1, label='Reported Results')
        ]
        
        ytick_labels = [label_lookup[exp] for exp in available_experiments]
        plt.yticks(y, ytick_labels)
        plt.xlabel(metric)
        plt.title(f"{metric} for {dataset_name} - {split}")
        plt.gca().invert_yaxis()
        # plt.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.show()

def plot_experiment_metrics_hbar_no_groups(dataset_name, experiments_dir, label_per_experiment, split, cbramod_experiments_dir=None, metrics_to_plot=None, figsize=(10, 6),  
                                           show_reported_cbramod=True, show_reported_labram=True, show_biot_reported=True, show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Plots mean performance metrics with standard deviation error bars for each experiment
    using horizontal bar charts.

    Parameters
    ----------
    base_dir : str
        Path containing experiment subdirectories. Each experiment must have
        `results/mean_performance_metrics.pkl` and `results/std_performance_metrics.pkl`.
    metrics_to_plot : list[str] or None
        Specific metrics to plot. If None, all metrics found in the first experiment are used.
    figsize : tuple
        Size of the matplotlib figure.
    """

    experiments, labels = zip(*label_per_experiment.items())

    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}

    available_experiments = []

    # Add reported experiments first
    available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, show_st_transformer_reported, show_eeg_conformer_reported)

    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue
        else:
            available_experiments.append(exp)

        total_params = exp_config['total_parameters']

        if split == 'val':
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        # Print mean and std
        print(f"Experiment: {exp}"
              f"\nMean Metrics: {mean_metrics}"
              f"\nStd Metrics: {std_metrics}\n")

        mean_data[exp] = mean_metrics
        std_data[exp] = std_metrics
        total_params_per_exp[exp] = total_params

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")

    # Determine metrics to plot
    all_metrics = list(next(iter(mean_data.values())).keys())
    if metrics_to_plot is None:
        metrics_to_plot = all_metrics

    # Create total_params_list with proper structure
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot each metric separately
    for metric in metrics_to_plot:
        if metric == 'confusion_matrix':
            continue
        means = [mean_data[exp][metric] for exp in available_experiments]
        stds = [std_data[exp][metric] for exp in available_experiments]

        tmp_available_experiments = available_experiments

        y = np.arange(len(tmp_available_experiments))

        plt.figure(figsize=figsize)
        bars = plt.barh(y, means, xerr=stds, capsize=5, alpha=0.7, color=colors[:len(tmp_available_experiments)])

        # Add parameter counts as text on each bar
        for i, (bar, params) in enumerate(zip(bars, total_params_list)):
            # Format parameter count (e.g., 1.2M, 4.0M, etc.)
            if params >= 1000000:
                param_text = f"{params/1000000:.1f}M"
            elif params >= 1000:
                param_text = f"{params/1000:.1f}K"
            else:
                param_text = str(params)

            # Position text at the end of the bar (or slightly inside if bar is very short)
            bar_width = bar.get_width()
            upper_ci = [m + s for m, s in zip(means, stds)]
            text_x = bar_width + (max(upper_ci) * -0.18)

            plt.text(text_x, bar.get_y() + bar.get_height()/2,
                    param_text,
                    ha='left', va='center',
                    fontweight='bold', fontsize=9)
            
        ytick_labels = [label_lookup[exp] for exp in available_experiments]

        plt.yticks(y, ytick_labels)
        plt.xlabel(metric)
        plt.title(f"{metric} for {dataset_name} - {split}")
        plt.gca().invert_yaxis()  # Highest experiment at top
        plt.show()

def print_performance_table(dataset_name, experiments_dir, label_per_experiment, split, 
                          desired_order, cbramod_experiments_dir=None, show_reported_cbramod=True, 
                          show_reported_labram=True, show_biot_reported=True, 
                          show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Prints a nicely formatted table of performance metrics per experiment.
    
    Args:
        dataset_name: Name of the dataset
        experiments_dir: Directory containing experiment results
        label_per_experiment: Dictionary mapping experiment names to labels
        split: 'val' or 'test' split to evaluate
        cbramod_experiments_dir: Directory for cbramod experiments (optional)
        show_reported_*: Boolean flags for including reported results
    """
    
    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    # Load metrics from each experiment
    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}
    problem_type = None

    available_experiments = []

    # Add reported experiments first (using your existing add_reported_data function)
    available_experiments, mean_data, std_data, _, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, 
        show_st_transformer_reported, show_eeg_conformer_reported)

    # Add experimental results
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue
        else:
            available_experiments.append(exp)
            is_reported[exp] = False

        # Get problem type from the first experimental config
        if problem_type is None:
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']

        total_params = exp_config['total_parameters']

        if split == 'val':
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        mean_data[exp] = mean_metrics
        std_data[exp] = std_metrics
        total_params_per_exp[exp] = total_params

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")
    
    total_params_list = [total_params_per_exp[exp] for exp in available_experiments]

    # Determine metric order based on problem type
    if problem_type == 'binary':
        metric_order = ['balanced_accuracy', 'pr_auc', 'roc_auc']
        metric_display_names = ['Balanced Accuracy', 'AUC-PR', 'AUROC']
    elif problem_type == 'multiclass':
        metric_order = ['balanced_accuracy', 'cohen_kappa', 'weighted_f1']
        metric_display_names = ['Balanced Accuracy', 'Cohen\'s Kappa', 'Weighted F1']
    else:
        # Fallback: use all available metrics
        all_metrics = list(next(iter(mean_data.values())).keys())
        metric_order = [m for m in all_metrics if m != 'confusion_matrix']
        metric_display_names = metric_order

    # Create the table data
    table_data = []
    
    for exp in available_experiments:
        row = {'Methods': label_lookup[exp]}
        row['#Params'] = f"{total_params_per_exp[exp]:,}"
        # row['#Params'] = total_params_per_exp[exp]
        
        for metric, display_name in zip(metric_order, metric_display_names):
            if metric in mean_data[exp]:
                mean_val = mean_data[exp][metric]
                std_val = std_data[exp][metric]
                # Format as mean ± std with 3 decimal places
                row[display_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[display_name] = "N/A"
        
        table_data.append(row)

    # Create DataFrame and print
    df = pd.DataFrame(table_data)
    
    # Sort the DataFrame by the desired order
    df_sorted = df.copy()
    if desired_order is not None:
        # Reorder the DataFrame according to desired_order
        # Create a mapping from method name to its index in the desired order
        method_order_map = {method: i for i, method in enumerate(desired_order)}
        df_sorted['order'] = df_sorted['Methods'].map(method_order_map)
        df_sorted = df_sorted.sort_values('order').drop('order', axis=1).reset_index(drop=True)
    
    # Print table header
    print(f"\nPerformance Results for {dataset_name} ({split} split)")
    return df_sorted

def get_performance_per_epoch(experiments_dir, label_per_experiment,eeg_conformer_experiments_dir=None, cbramod_experiments_dir=None):
    experiments, labels = zip(*label_per_experiment.items())

    # Load metrics from each experiment
    train_main_losses = {}
    train_balanced_accuracies = {}
    train_pr_aucs = {}
    train_roc_aucs = {}
    train_cohen_kappas = {}
    train_weighted_f1s = {}
    train_confusion_matrices = {}

    val_main_losses = {}
    val_balanced_accuracies = {}
    val_pr_aucs = {}
    val_roc_aucs = {}
    val_cohen_kappas = {}
    val_weighted_f1s = {}
    val_confusion_matrices = {}

    seed_set = set()

    # Add experimental results
    for exp, label in zip(experiments, labels):
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
                extra_info = pd.read_pickle(os.path.join(cbramod_experiments_dir, exp, "results", "extra_info_per_seed.pkl"))
            else:
                continue

        elif 'eegconformer' in exp.lower():
            if eeg_conformer_experiments_dir is not None:
                exp_config = load_experiment_config(os.path.join(eeg_conformer_experiments_dir, exp))
                extra_info = pd.read_pickle(os.path.join(eeg_conformer_experiments_dir, exp, "results", "extra_info_per_seed.pkl"))
            else:
                continue
        else:
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))
            extra_info = pd.read_pickle(os.path.join(experiments_dir, exp, "results", "extra_info_per_seed.pkl"))

        seed = list(extra_info.keys())[0]

        train_main_losses[label] = extra_info[seed]['train_main_losses']
        train_balanced_accuracies[label] = extra_info[seed]['train_balanced_acc_list']
        train_pr_aucs[label] = extra_info[seed]['train_pr_auc_list']
        train_roc_aucs[label] = extra_info[seed]['train_roc_auc_list']
        train_cohen_kappas[label] = extra_info[seed]['train_cohen_kappa_list']
        train_weighted_f1s[label] = extra_info[seed]['train_weighted_f1_list']
        train_confusion_matrices[label] = extra_info[seed]['train_cm_list']

        val_main_losses[label] = extra_info[seed]['val_main_losses']
        val_balanced_accuracies[label] = extra_info[seed]['val_balanced_acc_list']
        val_pr_aucs[label] = extra_info[seed]['val_pr_auc_list']
        val_roc_aucs[label] = extra_info[seed]['val_roc_auc_list']
        val_cohen_kappas[label] = extra_info[seed]['val_cohen_kappa_list']
        val_weighted_f1s[label] = extra_info[seed]['val_weighted_f1_list']
        val_confusion_matrices[label] = extra_info[seed]['val_cm_list']
        
        seed_set.add(seed)

    performance_dict = {
        'train_main_losses': train_main_losses,
        'train_balanced_accuracies': train_balanced_accuracies,
        'train_pr_aucs': train_pr_aucs,
        'train_roc_aucs': train_roc_aucs,
        'train_cohen_kappas': train_cohen_kappas,
        'train_weighted_f1s': train_weighted_f1s,
        'train_confusion_matrices': train_confusion_matrices,
        'val_main_losses': val_main_losses,
        'val_balanced_accuracies': val_balanced_accuracies,
        'val_pr_aucs': val_pr_aucs,
        'val_roc_aucs': val_roc_aucs,
        'val_cohen_kappas': val_cohen_kappas,
        'val_weighted_f1s': val_weighted_f1s,
        'val_confusion_matrices': val_confusion_matrices
    }

    assert len(seed_set) == 1, "All experiments must use the same seed for fair comparison."

    return performance_dict

def get_efficiency_stats(experiments_dir, label_per_experiment,eeg_conformer_experiments_dir=None, cbramod_experiments_dir=None):
    """
    Prints a nicely formatted table of efficiency stats per experiment.
    
    Args:
        dataset_name: Name of the dataset
        experiments_dir: Directory containing experiment results
        label_per_experiment: Dictionary mapping experiment names to labels
        split: 'val' or 'test' split to evaluate
        desired_order: Desired order of experiments in the output
        eeg_conformer_experiments_dir: Directory for eegconformer experiments (optional)
        cbramod_experiments_dir: Directory for cbramod experiments (optional)

    """
    
    experiments, labels = zip(*label_per_experiment.items())

    # Load metrics from each experiment
    total_params_per_exp = {}
    total_flops_per_exp = {}
    epoch_run_times_per_exp = {}
    inference_run_times_per_exp = {}

    seed_set = set()

    # Add experimental results
    for exp, label in zip(experiments, labels):
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
                extra_info = pd.read_pickle(os.path.join(cbramod_experiments_dir, exp, "results", "extra_info_per_seed.pkl"))
            else:
                continue

        elif 'eegconformer' in exp.lower():
            if eeg_conformer_experiments_dir is not None:
                exp_config = load_experiment_config(os.path.join(eeg_conformer_experiments_dir, exp))
                extra_info = pd.read_pickle(os.path.join(eeg_conformer_experiments_dir, exp, "results", "extra_info_per_seed.pkl"))
            else:
                continue
        else:
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))
            extra_info = pd.read_pickle(os.path.join(experiments_dir, exp, "results", "extra_info_per_seed.pkl"))
        seed = list(extra_info.keys())[0]
        epoch_run_times = extra_info[seed]['epoch_run_times']
        inference_run_times = extra_info[seed]['inference_run_times']

        total_params = exp_config['total_parameters']
        total_flops = exp_config['total_flops']

        total_params_per_exp[label] = total_params
        total_flops_per_exp[label] = total_flops
        epoch_run_times_per_exp[label] = epoch_run_times
        inference_run_times_per_exp[label] = inference_run_times
        
        seed_set.add(seed)

    assert len(seed_set) == 1, "All experiments must use the same seed for fair comparison."

    return total_params_per_exp, total_flops_per_exp, epoch_run_times_per_exp, inference_run_times_per_exp

def print_performance_table_separate_stats(dataset_name, experiments_dir, label_per_experiment, split, 
                                           desired_order, eeg_conformer_experiments_dir=None, cbramod_experiments_dir=None, show_reported_cbramod=True, 
                                           show_reported_labram=True, show_biot_reported=True, 
                                           show_st_transformer_reported=True, show_eeg_conformer_reported=True):
    """
    Same as print_performance_table but with separate mean and std columns for each metric.
    """
    experiments, labels = zip(*label_per_experiment.items())
    label_lookup = dict(label_per_experiment)

    mean_data = {}
    std_data = {}
    total_params_per_exp = {}
    is_reported = {}
    problem_type = None
    available_experiments = []

    # Add reported data
    available_experiments, mean_data, std_data, _, is_reported, labels, label_lookup = add_reported_data(
        available_experiments, mean_data, std_data, total_params_per_exp, is_reported, labels, label_lookup, dataset_name,
        show_reported_cbramod, show_reported_labram, show_biot_reported, 
        show_st_transformer_reported, show_eeg_conformer_reported)

    # Load experimental results
    for exp in experiments:
        if 'cbramod' in exp.lower():
            if cbramod_experiments_dir is not None:
                mean_path = os.path.join(cbramod_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(cbramod_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(cbramod_experiments_dir, exp))
            else:
                continue
        elif 'eegconformer' in exp.lower():
            if eeg_conformer_experiments_dir is not None:
                mean_path = os.path.join(eeg_conformer_experiments_dir, exp, "results", "mean_performance_metrics.pkl")
                std_path = os.path.join(eeg_conformer_experiments_dir, exp, "results", "std_performance_metrics.pkl")
                exp_config = load_experiment_config(os.path.join(eeg_conformer_experiments_dir, exp))
            else:
                continue
        else:
            mean_path = os.path.join(experiments_dir, exp, "results", "mean_performance_metrics.pkl")
            std_path = os.path.join(experiments_dir, exp, "results", "std_performance_metrics.pkl")
            exp_config = load_experiment_config(os.path.join(experiments_dir, exp))

        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            continue
        else:
            available_experiments.append(exp)
            is_reported[exp] = False

        if problem_type is None:
            problem_type = exp_config['modeling_approach_config']['params']['problem_type']

        total_params = exp_config['total_parameters']

        if split == 'val':
            mean_metrics, std_metrics = get_val_performance(experiments_dir, exp, problem_type)
        elif split == 'test':
            mean_metrics, std_metrics = get_test_performance(mean_path, std_path)

        mean_data[exp] = mean_metrics
        std_data[exp] = std_metrics
        total_params_per_exp[exp] = total_params

    if not mean_data:
        raise ValueError("No experiments with valid results found in the given directory.")
    
    # Define metric order
    if problem_type == 'binary':
        metric_order = ['balanced_accuracy', 'pr_auc', 'roc_auc']
        metric_display_names = ['Balanced Accuracy', 'AUC-PR', 'AUROC']
    elif problem_type == 'multiclass':
        metric_order = ['balanced_accuracy', 'cohen_kappa', 'weighted_f1']
        metric_display_names = ['Balanced Accuracy', 'Cohen\'s Kappa', 'Weighted F1']
    else:
        all_metrics = list(next(iter(mean_data.values())).keys())
        metric_order = [m for m in all_metrics if m != 'confusion_matrix']
        metric_display_names = metric_order

    # Build table with separate mean/std columns
    table_data = []
    for exp in available_experiments:
        row = {'Methods': label_lookup[exp]}
        row['#Params'] = f"{total_params_per_exp[exp]:,}"
        
        for metric, display_name in zip(metric_order, metric_display_names):
            if metric in mean_data[exp]:
                row[f"{display_name}_mean"] = mean_data[exp][metric]
                row[f"{display_name}_std"] = std_data[exp][metric]
            else:
                row[f"{display_name}_mean"] = None
                row[f"{display_name}_std"] = None
        
        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Apply desired ordering
    if desired_order is not None:
        order_map = {m: i for i, m in enumerate(desired_order)}
        df['order'] = df['Methods'].map(order_map)
        df = df.sort_values('order').drop(columns='order').reset_index(drop=True)

    print(f"\nPerformance Results for {dataset_name} ({split} split)")
    return df

def plot_for_final_paper(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None
):
    color_palette = sns.color_palette("Set1")
    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI (2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic (2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a (4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI (4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016 (2-Class, 7,143 Samples)',
        'seedv': 'SEED-V (5-Class, 117,744 Samples)',
        'tuev': 'TUEV (6-Class, 113,353 Samples)',
        'faced': 'FACED (9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for each set of metrics
    def calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names):
        """Calculate global y-limits for each metric, handling shared metrics correctly"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for Balanced Accuracy across ALL datasets (since it's shared)
        balanced_acc_mins = []
        balanced_acc_maxs = []
        
        for dataset_name in dataset_names:  # All datasets
            df = df_per_dataset[dataset_name]
            
            for val in df['Balanced Accuracy']:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    balanced_acc_mins.append(mean - std)
                    balanced_acc_maxs.append(mean + std)
        
        # Set global limits for Balanced Accuracy
        y_min = max(0, min(balanced_acc_mins) - 0.02)
        y_max = min(1, max(balanced_acc_maxs) + 0.02)
        global_limits['Balanced Accuracy'] = (y_min, y_max)
        
        # Calculate limits for binary-only metrics (AUC-PR, AUROC)
        if binary_datasets:
            binary_only_metrics = ['AUC-PR', 'AUROC']
            for metric in binary_only_metrics:
                all_mins = []
                all_maxs = []
                
                for dataset_name in binary_datasets:
                    df = df_per_dataset[dataset_name]
                    
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        # Calculate limits for multiclass-only metrics (Cohen's Kappa, Weighted F1)
        if multiclass_datasets:
            multiclass_only_metrics = ["Cohen's Kappa", 'Weighted F1']
            for metric in multiclass_only_metrics:
                all_mins = []
                all_maxs = []
                
                for dataset_name in multiclass_datasets:
                    df = df_per_dataset[dataset_name]
                    
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits by metric type
    global_ylimits = calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names)

    num_metrics = 3
    bar_width = 1

    fig = plt.figure(figsize=(15, 6))
    # Main 2x2 grid for datasets
    outer = GridSpec(2, 2, figure=fig, wspace=0.13, hspace=0.3)  
    # ↑ Adjust wspace and hspace for dataset spacing

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        
        x = np.arange(len(df['Methods']))

        if "Cohen\'s Kappa" in df.columns:
            metrics_to_plot = ['Balanced Accuracy', "Cohen's Kappa", 'Weighted F1']
        else:
            metrics_to_plot = ['Balanced Accuracy', 'AUC-PR', 'AUROC']
        
        # Find row and col in the outer 2x2 grid
        row = dataset_idx // 2
        col = dataset_idx % 2

        # Inner grid for metrics (1x3 inside each dataset cell)
        inner = outer[row, col].subgridspec(1, num_metrics, wspace=0.31)  
        # ↑ Adjust wspace for spacing between metrics

        for i, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(inner[0, i])

            means = []
            stds = []
            for val in df[metric]:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    means.append(mean)
                    stds.append(std)
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x, means, yerr=stds, capsize=5, 
                        color=sns.color_palette(color_palette, len(df['Methods'])), 
                        width=bar_width)
            ax.set_xticks(x)
            ax.set_title(metric, fontsize=12)

            # Use global y-limits for this metric
            y_min, y_max = global_ylimits[metric]
            ax.set_ylim(y_min, y_max)

            # Force exactly 3 ticks
            ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

            # Format tick labels to 2 decimal places
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # Remove x-tick labels to save space
            ax.set_xticklabels([])
            
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])   # removes ticks *and* labels
        
        # Add dataset title across the row of metrics
        fig.text(
            inner[0, 1].get_position(fig).x0 + inner[0, 1].get_position(fig).width/2,
            inner[0, 0].get_position(fig).y1 + 0.05,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    colors = sns.color_palette(color_palette, len(df_first['Methods']))
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, bbox_to_anchor=(0.5, 1.03) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_grayscale(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None
):
    num_methods = len(list(df_per_dataset.values())[0]['Methods'])
    # Generate grayscale colors from dark to light with good separation
    grayscale_colors = []
    for i in range(num_methods):
        # Create values from 0.2 (dark gray) to 0.8 (light gray)
        gray_value = 0.2 + (0.6 * i / max(1, num_methods - 1))
        grayscale_colors.append((gray_value, gray_value, gray_value))

    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI (2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic (2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a (4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI (4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016 (2-Class, 7,143 Samples)',
        'seedv': 'SEED-V (5-Class, 117,744 Samples)',
        'tuev': 'TUEV (6-Class, 113,353 Samples)',
        'faced': 'FACED (9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for each set of metrics
    def calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names):
        """Calculate global y-limits for each metric, handling shared metrics correctly"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for Balanced Accuracy across ALL datasets (since it's shared)
        balanced_acc_mins = []
        balanced_acc_maxs = []
        
        for dataset_name in dataset_names:  # All datasets
            df = df_per_dataset[dataset_name]
            
            for val in df['Balanced Accuracy']:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    balanced_acc_mins.append(mean - std)
                    balanced_acc_maxs.append(mean + std)
        
        # Set global limits for Balanced Accuracy
        y_min = max(0, min(balanced_acc_mins) - 0.02)
        y_max = min(1, max(balanced_acc_maxs) + 0.02)
        global_limits['Balanced Accuracy'] = (y_min, y_max)
        
        # Calculate limits for binary-only metrics (AUC-PR, AUROC)
        if binary_datasets:
            binary_only_metrics = ['AUC-PR', 'AUROC']
            for metric in binary_only_metrics:
                all_mins = []
                all_maxs = []
                
                for dataset_name in binary_datasets:
                    df = df_per_dataset[dataset_name]
                    
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        # Calculate limits for multiclass-only metrics (Cohen's Kappa, Weighted F1)
        if multiclass_datasets:
            multiclass_only_metrics = ["Cohen's Kappa", 'Weighted F1']
            for metric in multiclass_only_metrics:
                all_mins = []
                all_maxs = []
                
                for dataset_name in multiclass_datasets:
                    df = df_per_dataset[dataset_name]
                    
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits by metric type
    global_ylimits = calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names)

    num_metrics = 3
    bar_width = 1

    fig = plt.figure(figsize=(15, 6))
    # Main 2x2 grid for datasets
    outer = GridSpec(2, 2, figure=fig, wspace=0.13, hspace=0.3)  
    # ↑ Adjust wspace and hspace for dataset spacing

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        
        x = np.arange(len(df['Methods']))

        if "Cohen\'s Kappa" in df.columns:
            metrics_to_plot = ['Balanced Accuracy', "Cohen's Kappa", 'Weighted F1']
        else:
            metrics_to_plot = ['Balanced Accuracy', 'AUC-PR', 'AUROC']
        
        # Find row and col in the outer 2x2 grid
        row = dataset_idx // 2
        col = dataset_idx % 2

        # Inner grid for metrics (1x3 inside each dataset cell)
        inner = outer[row, col].subgridspec(1, num_metrics, wspace=0.31)  
        # ↑ Adjust wspace for spacing between metrics

        for i, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(inner[0, i])

            means = []
            stds = []
            for val in df[metric]:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    means.append(mean)
                    stds.append(std)
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x, means, yerr=stds, capsize=5, 
                        color=grayscale_colors, 
                        width=bar_width,
                        edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_title(metric, fontsize=12)

            # Use global y-limits for this metric
            y_min, y_max = global_ylimits[metric]
            ax.set_ylim(y_min, y_max)

            # Force exactly 3 ticks
            ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

            # Format tick labels to 2 decimal places
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # Remove x-tick labels to save space
            ax.set_xticklabels([])
            
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])   # removes ticks *and* labels
        
        # Add dataset title across the row of metrics
        fig.text(
            inner[0, 1].get_position(fig).x0 + inner[0, 1].get_position(fig).width/2,
            inner[0, 0].get_position(fig).y1 + 0.05,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=grayscale_colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, bbox_to_anchor=(0.5, 1.03) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}-gray.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_1x4(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None
):
    color_palette = sns.color_palette("Set1")
    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI\n(2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic\n(2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a\n(4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI\n(4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016\n(2-Class, 7,143 Samples)',
        'seedv': 'SEED-V\n(5-Class, 117,744 Samples)',
        'tuev': 'TUEV\n(6-Class, 113,353 Samples)',
        'faced': 'FACED\n(9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for AUROC and Cohen's Kappa
    def calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names):
        """Calculate global y-limits for AUROC and Cohen's Kappa"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for AUROC (binary datasets)
        if binary_datasets:
            auroc_mins = []
            auroc_maxs = []
            
            for dataset_name in binary_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df['AUROC']:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        auroc_mins.append(mean - std)
                        auroc_maxs.append(mean + std)
            
            y_min = max(0, min(auroc_mins) - 0.02)
            y_max = min(1, max(auroc_maxs) + 0.02)
            global_limits['AUROC'] = (y_min, y_max)
        
        # Calculate limits for Cohen's Kappa (multiclass datasets)
        if multiclass_datasets:
            kappa_mins = []
            kappa_maxs = []
            
            for dataset_name in multiclass_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df["Cohen's Kappa"]:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        kappa_mins.append(mean - std)
                        kappa_maxs.append(mean + std)
            
            y_min = max(0, min(kappa_mins) - 0.02)
            y_max = min(1, max(kappa_maxs) + 0.02)
            global_limits["Cohen's Kappa"] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits for selected metrics
    global_ylimits = calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names)

    bar_width = 1

    # Create 1x4 figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        ax = axes[dataset_idx]
        
        x = np.arange(len(df['Methods']))

        # Determine which metric to plot based on dataset type
        if "Cohen\'s Kappa" in df.columns:
            metric_to_plot = "Cohen's Kappa"
        else:
            metric_to_plot = 'AUROC'
        
        # Extract means and standard deviations
        means = []
        stds = []
        for val in df[metric_to_plot]:
            if val != "N/A":
                mean, std = map(float, val.split(' ± '))
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bar plot
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                    color=sns.color_palette(color_palette, len(df['Methods'])), 
                    width=bar_width)
        
        # Set x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])  # Remove x-tick labels to save space
        
        # Set title (dataset name)
        ax.set_title(metric_to_plot, fontsize=12)

        fig.text(
            ax.get_position().x0 + ax.get_position().width/2,
            ax.get_position().y1 + 0.12,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )
        
        # Use global y-limits for this metric
        if metric_to_plot in global_ylimits:
            y_min, y_max = global_ylimits[metric_to_plot]
            ax.set_ylim(y_min, y_max)

        # Force exactly 3 ticks on y-axis
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

        # Format tick labels to 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Add grid
        ax.grid(True, alpha=0.3)

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    colors = sns.color_palette(color_palette, len(df_first['Methods']))
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, 
            bbox_to_anchor=(0.5, 1.05) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}_1x4.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_1x4_grayscale(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None
):
    num_methods = len(list(df_per_dataset.values())[0]['Methods'])
    # Generate grayscale colors from dark to light with good separation
    grayscale_colors = []
    for i in range(num_methods):
        # Create values from 0.2 (dark gray) to 0.8 (light gray)
        gray_value = 0.2 + (0.6 * i / max(1, num_methods - 1))
        grayscale_colors.append((gray_value, gray_value, gray_value))

    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI\n(2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic\n(2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a\n(4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI\n(4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016\n(2-Class, 7,143 Samples)',
        'seedv': 'SEED-V\n(5-Class, 117,744 Samples)',
        'tuev': 'TUEV\n(6-Class, 113,353 Samples)',
        'faced': 'FACED\n(9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for AUROC and Cohen's Kappa
    def calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names):
        """Calculate global y-limits for AUROC and Cohen's Kappa"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for AUROC (binary datasets)
        if binary_datasets:
            auroc_mins = []
            auroc_maxs = []
            
            for dataset_name in binary_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df['AUROC']:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        auroc_mins.append(mean - std)
                        auroc_maxs.append(mean + std)
            
            y_min = max(0, min(auroc_mins) - 0.02)
            y_max = min(1, max(auroc_maxs) + 0.02)
            global_limits['AUROC'] = (y_min, y_max)
        
        # Calculate limits for Cohen's Kappa (multiclass datasets)
        if multiclass_datasets:
            kappa_mins = []
            kappa_maxs = []
            
            for dataset_name in multiclass_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df["Cohen's Kappa"]:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        kappa_mins.append(mean - std)
                        kappa_maxs.append(mean + std)
            
            y_min = max(0, min(kappa_mins) - 0.02)
            y_max = min(1, max(kappa_maxs) + 0.02)
            global_limits["Cohen's Kappa"] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits for selected metrics
    global_ylimits = calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names)

    bar_width = 1

    # Create 1x4 figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        ax = axes[dataset_idx]
        
        x = np.arange(len(df['Methods']))

        # Determine which metric to plot based on dataset type
        if "Cohen\'s Kappa" in df.columns:
            metric_to_plot = "Cohen's Kappa"
        else:
            metric_to_plot = 'AUROC'
        
        # Extract means and standard deviations
        means = []
        stds = []
        for val in df[metric_to_plot]:
            if val != "N/A":
                mean, std = map(float, val.split(' ± '))
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bar plot
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                    color=grayscale_colors,
                    width=bar_width)
        
        # Set x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])  # Remove x-tick labels to save space
        
        # Set title (dataset name)
        ax.set_title(metric_to_plot, fontsize=12)

        fig.text(
            ax.get_position().x0 + ax.get_position().width/2,
            ax.get_position().y1 + 0.12,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )
        
        # Use global y-limits for this metric
        if metric_to_plot in global_ylimits:
            y_min, y_max = global_ylimits[metric_to_plot]
            ax.set_ylim(y_min, y_max)

        # Force exactly 3 ticks on y-axis
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

        # Format tick labels to 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Add grid
        ax.grid(True, alpha=0.3)

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=grayscale_colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, 
            bbox_to_anchor=(0.5, 1.05) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}_1x4-gray.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_for_final_paper_2x2(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None,
    plot_type='bar'  # 'bar' or 'line'
):
    color_palette = sns.color_palette("Set1")
    color_palette[5] = sns.color_palette(["#17becf"])[0]
    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI\n(2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic\n(2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a\n(4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI\n(4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016\n(2-Class, 7,143 Samples)',
        'seedv': 'SEED-V\n(5-Class, 117,744 Samples)',
        'tuev': 'TUEV\n(6-Class, 113,353 Samples)',
        'faced': 'FACED\n(9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for AUROC and Cohen's Kappa
    def calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names):
        """Calculate global y-limits for AUROC and Cohen's Kappa"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for AUROC (binary datasets)
        if binary_datasets:
            auroc_mins = []
            auroc_maxs = []
            
            for dataset_name in binary_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df['AUROC']:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        auroc_mins.append(mean - std)
                        auroc_maxs.append(mean + std)
            
            y_min = max(0, min(auroc_mins) - 0.02)
            y_max = min(1, max(auroc_maxs) + 0.02)
            global_limits['AUROC'] = (y_min, y_max)
        
        # Calculate limits for Cohen's Kappa (multiclass datasets)
        if multiclass_datasets:
            kappa_mins = []
            kappa_maxs = []
            
            for dataset_name in multiclass_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df["Cohen's Kappa"]:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        kappa_mins.append(mean - std)
                        kappa_maxs.append(mean + std)
            
            y_min = max(0, min(kappa_mins) - 0.02)
            y_max = min(1, max(kappa_maxs) + 0.02)
            global_limits["Cohen's Kappa"] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits for selected metrics
    global_ylimits = calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names)

    bar_width = 1

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.55)  # Adjust spacing between subplots

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        # Calculate row and column indices for 2x2 layout
        row_idx = dataset_idx // 2
        col_idx = dataset_idx % 2
        ax = axes[row_idx, col_idx]
        
        x = np.arange(len(df['Methods']))

        # Determine which metric to plot based on dataset type
        if "Cohen\'s Kappa" in df.columns:
            metric_to_plot = "Cohen's Kappa"
        else:
            metric_to_plot = 'AUROC'
        
        # Extract means and standard deviations
        means = []
        stds = []
        for val in df[metric_to_plot]:
            if val != "N/A":
                mean, std = map(float, val.split(' ± '))
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bar plot
        if plot_type == 'bar':
            bars = ax.bar(x, means, yerr=stds, capsize=5, 
                        color=sns.color_palette(color_palette, len(df['Methods'])), 
                        width=bar_width)
        elif plot_type == 'line':
            ax.errorbar(x, means, yerr=stds, capsize=5, 
                        color='black', marker='o', linestyle='-', 
                        markersize=8, linewidth=2)
        else:
            raise ValueError("plot_type must be 'bar' or 'line'")
            
        
        # Set x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])  # Remove x-tick labels to save space
        
        # Set title (metric name and dataset name)
        ax.set_title(metric_to_plot, fontsize=12)
        
        fig.text(
            ax.get_position().x0 + ax.get_position().width/2,
            ax.get_position().y1 + 0.07,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )
        
        # Use global y-limits for this metric
        if metric_to_plot in global_ylimits:
            y_min, y_max = global_ylimits[metric_to_plot]
            ax.set_ylim(y_min, y_max)

        # Force exactly 3 ticks on y-axis
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

        # Format tick labels to 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Add grid
        ax.grid(True, alpha=0.3)

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    colors = sns.color_palette(color_palette, len(df_first['Methods']))
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, 
            bbox_to_anchor=(0.5, 0.98) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}_{plot_type}_2x2.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_1x2(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    ncol=None,
    loc=None,
    plot_type='bar', # 'bar' or 'line',
    figsize=(10, 5),
    dataset_name_y_offset=0.07
):
    color_palette = sns.color_palette("Set1")
    color_palette[5] = sns.color_palette(["#17becf"])[0]
    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 2, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI\n(2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic\n(2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a\n(4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI\n(4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016\n(2-Class, 7,143 Samples)',
        'seedv': 'SEED-V\n(5-Class, 117,744 Samples)',
        'tuev': 'TUEV\n(6-Class, 113,353 Samples)',
        'faced': 'FACED\n(9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for AUROC and Cohen's Kappa
    def calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names):
        """Calculate global y-limits for AUROC and Cohen's Kappa"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for AUROC (binary datasets)
        if binary_datasets:
            auroc_mins = []
            auroc_maxs = []
            
            for dataset_name in binary_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df['AUROC']:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        auroc_mins.append(mean - std)
                        auroc_maxs.append(mean + std)
            
            y_min = max(0, min(auroc_mins) - 0.02)
            y_max = min(1, max(auroc_maxs) + 0.02)
            global_limits['AUROC'] = (y_min, y_max)
        
        # Calculate limits for Cohen's Kappa (multiclass datasets)
        if multiclass_datasets:
            kappa_mins = []
            kappa_maxs = []
            
            for dataset_name in multiclass_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df["Cohen's Kappa"]:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        kappa_mins.append(mean - std)
                        kappa_maxs.append(mean + std)
            
            y_min = max(0, min(kappa_mins) - 0.02)
            y_max = min(1, max(kappa_maxs) + 0.02)
            global_limits["Cohen's Kappa"] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits for selected metrics
    global_ylimits = calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names)

    bar_width = 1

    # Create 2x2 figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, hspace=0.55)  # Adjust spacing between subplots

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        # Calculate row and column indices for 2x2 layout
        col_idx = dataset_idx % 2
        ax = axes[col_idx]
        
        x = np.arange(len(df['Methods']))

        # Determine which metric to plot based on dataset type
        if "Cohen\'s Kappa" in df.columns:
            metric_to_plot = "Cohen's Kappa"
        else:
            metric_to_plot = 'AUROC'
        
        # Extract means and standard deviations
        means = []
        stds = []
        for val in df[metric_to_plot]:
            if val != "N/A":
                mean, std = map(float, val.split(' ± '))
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bar plot
        if plot_type == 'bar':
            bars = ax.bar(x, means, yerr=stds, capsize=5, 
                        color=sns.color_palette(color_palette, len(df['Methods'])), 
                        width=bar_width)
        elif plot_type == 'line':
            ax.errorbar(x, means, yerr=stds, capsize=5, 
                        color='black', marker='o', linestyle='-', 
                        markersize=8, linewidth=2)
        else:
            raise ValueError("plot_type must be 'bar' or 'line'")
            
        
        # Set x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])  # Remove x-tick labels to save space
        
        # Set title (metric name and dataset name)
        ax.set_title(metric_to_plot, fontsize=12)
        
        fig.text(
            ax.get_position().x0 + ax.get_position().width/2,
            ax.get_position().y1 + dataset_name_y_offset,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )
        
        # Use global y-limits for this metric
        if metric_to_plot in global_ylimits:
            y_min, y_max = global_ylimits[metric_to_plot]
            ax.set_ylim(y_min, y_max)

        # Force exactly 3 ticks on y-axis
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))

        # Format tick labels to 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Add grid
        ax.grid(True, alpha=0.3)

    # Create a single legend for all subplots at the top
    df_first = df_per_dataset[dataset_names[0]]
    handles = []
    colors = sns.color_palette(color_palette, len(df_first['Methods']))
    for i, method in enumerate(df_first['Methods']):
        handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))

    fig.legend(handles, df_first['Methods'], loc='upper center', 
            ncol=len(df_first['Methods']) if ncol is None else ncol, 
            bbox_to_anchor=(0.5, 0.98) if loc is None else loc, fontsize=12)

    plt.savefig(f'final_figures/{split}_{comparison_name}_{plot_type}_2x2.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_2x2_line_log(
    df_per_dataset,
    split,
    comparison_name,
    dataset_names,
    x_value_dict=None,
    x_axis_dict=None,
    use_log_xscale=False,
    use_param_count_as_x=False
):
    color_palette = sns.color_palette("Set1")
    # Get dataset names (assuming df_per_dataset is a dictionary with 4 datasets)
    if dataset_names is None:
        dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 4, "This code expects exactly 4 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI\n(2-Class, 11,988 Samples)',
        'stress': 'MentalArithmetic\n(2-Class, 1,707 Samples)',
        'bciciv2a': 'BCIC-IV-2a\n(4-Class, 5,088 Samples)',
        'physio': 'PhysioNet-MI\n(4-Class, 9,837 Samples)',
        'mumtaz': 'Mumtaz2016\n(2-Class, 7,143 Samples)',
        'seedv': 'SEED-V\n(5-Class, 117,744 Samples)',
        'tuev': 'TUEV\n(6-Class, 113,353 Samples)',
        'faced': 'FACED\n(9-Class, 10,332 Samples)',
    }

    # Pre-calculate global y-limits for AUROC and Cohen's Kappa
    def calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names):
        """Calculate global y-limits for AUROC and Cohen's Kappa"""
        
        # Separate datasets by metric type
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Calculate limits for AUROC (binary datasets)
        if binary_datasets:
            auroc_mins = []
            auroc_maxs = []
            
            for dataset_name in binary_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df['AUROC']:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        auroc_mins.append(mean - std)
                        auroc_maxs.append(mean + std)
            
            y_min = max(0, min(auroc_mins) - 0.02)
            y_max = min(1, max(auroc_maxs) + 0.02)
            global_limits['AUROC'] = (y_min, y_max)
        
        # Calculate limits for Cohen's Kappa (multiclass datasets)
        if multiclass_datasets:
            kappa_mins = []
            kappa_maxs = []
            
            for dataset_name in multiclass_datasets:
                df = df_per_dataset[dataset_name]
                
                for val in df["Cohen's Kappa"]:
                    if val != "N/A":
                        mean, std = map(float, val.split(' ± '))
                        kappa_mins.append(mean - std)
                        kappa_maxs.append(mean + std)
            
            y_min = max(0, min(kappa_mins) - 0.02)
            y_max = min(1, max(kappa_maxs) + 0.02)
            global_limits["Cohen's Kappa"] = (y_min, y_max)
        
        return global_limits

    # Calculate global y-limits for selected metrics
    global_ylimits = calculate_global_ylimits_for_selected_metrics(df_per_dataset, dataset_names)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.7)

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        row_idx = dataset_idx // 2
        col_idx = dataset_idx % 2
        ax = axes[row_idx, col_idx]
        
        # Create x values (log scale). The dict is label: value
        if x_value_dict and dataset_name in x_value_dict:
            x = [x_value_dict[dataset_name][method] for method in df['Methods']]
        else:
            x = np.arange(len(df['Methods']))  

        if use_param_count_as_x:
            # Get the param counts from the df #Params column
            param_counts = []
            for param_count in df['#Params']:
                if isinstance(param_count, str):
                    # Format like "1.2M", currently they look like "1,302,038"
                    param_count = int(param_count.replace(',', ''))
                    if param_count >= 1e6:
                        param_count = f"{param_count/1e6:.1f}M"
                    elif param_count >= 1e3:
                        param_count = f"{int(param_count/1e3)}K"
                    else:
                        param_count = str(param_count)
                param_counts.append(param_count)
            x = list(range(len(param_counts)))

        if "Cohen\'s Kappa" in df.columns:
            metric_to_plot = "Cohen's Kappa"
        else:
            metric_to_plot = 'AUROC'
        
        means = []
        stds = []
        for val in df[metric_to_plot]:
            if val != "N/A":
                mean, std = map(float, val.split(' ± '))
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)
        
        # Create line plot
        colors = sns.color_palette(color_palette, len(df['Methods']))
        for i in range(len(x)):
            ax.errorbar(x[i], means[i], yerr=stds[i], capsize=5, 
                        color=colors[i], marker='o', linestyle='', 
                        markersize=4, linewidth=2)
        
        # Connect points with line
        ax.plot(x, means, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
        
        if x_axis_dict is None:
            if use_log_xscale:
                ax.set_xscale('log')
            ax.set_xticks(x)
        else:
            # The keys are the locations and the values are the labels
            ax.set_xticks(list(x_axis_dict.keys()))
            ax.set_xticklabels(list(x_axis_dict.values()))

        if use_param_count_as_x:
            ax.set_xticks(x)
            ax.set_xticklabels(param_counts)
        else:
            # The keys are the locations and the values are the labels
            ax.set_xticks(list(x_axis_dict.keys()))
            ax.set_xticklabels(list(x_axis_dict.values()))
        
        ax.set_title(metric_to_plot, fontsize=12)
        
        fig.text(
            ax.get_position().x0 + ax.get_position().width/2,
            ax.get_position().y1 + 0.07,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )
        
        if metric_to_plot in global_ylimits:
            y_min, y_max = global_ylimits[metric_to_plot]
            ax.set_ylim(y_min, y_max)

        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        ax.grid(True, alpha=0.3)

    plt.savefig(f'final_figures/{split}_{comparison_name}_2x2_line.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_for_final_paper_2_datasets(
    df_per_dataset,
    split,
    comparison_name
):
    color_palette = sns.color_palette("Set1")
    # Get dataset names (assuming df_per_dataset is a dictionary with 2 datasets)
    dataset_names = list(df_per_dataset.keys())
    assert len(dataset_names) == 2, "This code expects exactly 2 datasets"

    proper_dataset_name_map = {
        'shu': 'SHU-MI (2-Class, 11,988 Samples)',
        'physio': 'PhysioNet-MI (4-Class, 9,837 Samples)',
        'bciciv2a': 'BCIC-IV-2a (4-Class, 5,088 Samples)',
        'stress': 'MentalArithmetic (2-Class, 1,707 Samples)',
    }

    # Pre-calculate global y-limits for each set of metrics
    def calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names):
        binary_datasets = []
        multiclass_datasets = []
        
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            if "Cohen\'s Kappa" in df.columns:
                multiclass_datasets.append(dataset_name)
            else:
                binary_datasets.append(dataset_name)
        
        global_limits = {}
        
        # Balanced Accuracy across all datasets
        balanced_acc_mins, balanced_acc_maxs = [], []
        for dataset_name in dataset_names:
            df = df_per_dataset[dataset_name]
            for val in df['Balanced Accuracy']:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    balanced_acc_mins.append(mean - std)
                    balanced_acc_maxs.append(mean + std)
        
        y_min = max(0, min(balanced_acc_mins) - 0.02)
        y_max = min(1, max(balanced_acc_maxs) + 0.02)
        global_limits['Balanced Accuracy'] = (y_min, y_max)
        
        # Binary-only metrics
        if binary_datasets:
            for metric in ['AUC-PR', 'AUROC']:
                all_mins, all_maxs = [], []
                for dataset_name in binary_datasets:
                    df = df_per_dataset[dataset_name]
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        # Multiclass-only metrics
        if multiclass_datasets:
            for metric in ["Cohen's Kappa", 'Weighted F1']:
                all_mins, all_maxs = [], []
                for dataset_name in multiclass_datasets:
                    df = df_per_dataset[dataset_name]
                    for val in df[metric]:
                        if val != "N/A":
                            mean, std = map(float, val.split(' ± '))
                            all_mins.append(mean - std)
                            all_maxs.append(mean + std)
                y_min = max(0, min(all_mins) - 0.02)
                y_max = min(1, max(all_maxs) + 0.02)
                global_limits[metric] = (y_min, y_max)
        
        return global_limits

    global_ylimits = calculate_global_ylimits_by_metric_type(df_per_dataset, dataset_names)

    num_metrics = 3
    bar_width = 1

    fig = plt.figure(figsize=(15, 3))
    # 1x2 grid for 2 datasets
    outer = GridSpec(1, 2, figure=fig, wspace=0.13, hspace=0.3)

    for dataset_idx, dataset_name in enumerate(dataset_names):
        df = df_per_dataset[dataset_name]
        x = np.arange(len(df['Methods']))

        if "Cohen\'s Kappa" in df.columns:
            metrics_to_plot = ['Balanced Accuracy', "Cohen's Kappa", 'Weighted F1']
        else:
            metrics_to_plot = ['Balanced Accuracy', 'AUC-PR', 'AUROC']

        # Inner grid for metrics (1x3 inside each dataset cell)
        inner = outer[0, dataset_idx].subgridspec(1, num_metrics, wspace=0.31)

        for i, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(inner[0, i])

            means, stds = [], []
            for val in df[metric]:
                if val != "N/A":
                    mean, std = map(float, val.split(' ± '))
                    means.append(mean)
                    stds.append(std)
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(
                x, means, yerr=stds, capsize=5,
                color=sns.color_palette(color_palette, len(df['Methods'])),
                width=bar_width
            )
            ax.set_xticks(x)
            ax.set_title(metric, fontsize=12)

            y_min, y_max = global_ylimits[metric]
            ax.set_ylim(y_min, y_max)

            ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            ax.set_xticklabels([])
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])

        # Add dataset title across metrics
        fig.text(
            inner[0, 1].get_position(fig).x0 + inner[0, 1].get_position(fig).width / 2,
            inner[0, 0].get_position(fig).y1 + 0.1,
            proper_dataset_name_map.get(dataset_name, dataset_name),
            fontsize=14, fontweight='bold', ha='center'
        )

    # Shared legend
    df_first = df_per_dataset[dataset_names[0]]
    colors = sns.color_palette(color_palette, len(df_first['Methods']))
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i, _ in enumerate(df_first['Methods'])]

    fig.legend(
        handles, df_first['Methods'], loc='upper center',
        ncol=len(df_first['Methods']), bbox_to_anchor=(0.5, 1.2), fontsize=12
    )

    plt.savefig(f'final_figures/{split}_{comparison_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
