"""
Lorem
"""

from stamp.local import local_config
import pandas as pd
import numpy as np
import subprocess
import datetime
import re
import torch
import random
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scale_features(
    features:pd.DataFrame
    )->pd.DataFrame:
    """Scales features to have mean of 0 and std of 1.

    Parameters
    ----------
    features : pd.DataFrame
        A dataframe mapping ids to a set of features. For examples,
        these features could be eeg features.

    Returns
    -------
    pd.DataFrame
        A dataframe mapping ids to their scaled features.
    """
    ids = features.index
    columns = features.columns

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features, index=ids, columns=columns)
    return features

def transform_features_with_pca(
    features:pd.DataFrame,
    top_or_bottom:str,
    desired_variance:float
    )->pd.DataFrame:
    """Runs PCA on the given features and automatically extracts the
    principal components (PCs) that cover either the top or bottom desired
    variance. For example, if you wish to get the PCs that cover the
    top 90% of the explained variance, then use top_or_bottom = 'top' and
    desired_variance = 0.90.

    Parameters
    ----------
    features : pd.DataFrame
        A dataframe mapping ids to a set of features. For examples,
        these features could be eeg features.

    top_or_bottom : str
        A string indicating whether to determine the PCs by
        the top variance or bottom variance.

    desired_variance : float
        A float [0,1] that determines how much variance
        the extracted PCs cover.

    Returns
    -------
    pd.DataFrame
        A dataframe mapping ids to the specific PCs
        that cover a percentage of variance.
    """
    assert 0 <= desired_variance <= 1, "The given value for desired_variance is not within the correct interval [0,1]."

    ids = features.index

    if top_or_bottom == 'top':
        pca = PCA(n_components=desired_variance, svd_solver='auto')
        pca.fit(features)
        features = pca.transform(features)
    elif top_or_bottom == 'bottom':
        pca = PCA().fit(features)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components_bottom_percent = np.argmax(cumulative_variance_ratio >= (1 - desired_variance)) + 1
        features = pca.transform(features)[:, n_components_bottom_percent:]
    else:
        raise ValueError('The given value for top_or_bottom is invalid. It should be top or bottom.')

    features = pd.DataFrame(features, index=ids)
    return features

def randomize_labels(
    labels
    ):
    randomized_labels = labels.sample(frac=1, random_state=42).reset_index(drop=True)
    randomized_labels.index = labels.index
    return randomized_labels

def natural_sort(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def fast_convert_dict_with_df_values_to_single_df(
    features_dict: dict
    ) -> pd.DataFrame:
    rows = []
    index_names = None
    col_names = None

    for i, (idd, df) in enumerate(features_dict.items()):
        if i == 0:
            # Save the index and column names from the first DF
            index_names = df.index.names if df.index.nlevels > 1 else [df.index.name or "index"]
            col_names = df.columns.tolist()

        index_tuples = df.index.to_numpy()
        values = df.values  # shape: (n_rows, n_columns)

        # Create rows: (idd, *index_values, *data_values)
        rows.extend([(idd, *idx, *vals) for idx, vals in zip(index_tuples, values)])

    full_column_names = ["id"] + index_names + col_names
    combined_df = pd.DataFrame(rows, columns=full_column_names)
    combined_df.set_index(["id"] + index_names, inplace=True)

    return combined_df

def convert_dict_with_df_values_to_single_df(
    features_dict:dict
    ):
    if type(features_dict[list(features_dict.keys())[0]]) == pd.DataFrame:
        columns = list(features_dict[list(features_dict.keys())[0]].columns)

        # Check for duplicate column names
        if len(columns) != len(set(columns)):
            raise Warning('There are duplicate column names in the single row DFs. This can cause an issue where the output '\
                        'dataframe may have an incorrect number of columns.')

        # Concatenate the DataFrames in the dictionary
        features_df = pd.concat(features_dict.values(), keys=features_dict.keys())

        # Check if there are duplicate index levels, ie index 1 has the exact same values per row as index 2
        if isinstance(features_df.index, pd.MultiIndex):
            idx = features_df.index
            n_levels = idx.nlevels
            to_keep = [True] * n_levels  # Track which levels to keep

            # Compare each pair of levels
            for i in range(n_levels):
                for j in range(i + 1, n_levels):
                    if to_keep[j] and (idx.get_level_values(i).equals(idx.get_level_values(j))):
                        to_keep[j] = False  # Mark redundant level for removal

            # Keep only unique levels
            if not all(to_keep):
                new_index = pd.MultiIndex.from_arrays(
                    [idx.get_level_values(i) for i in range(n_levels) if to_keep[i]],
                    names=[idx.names[i] for i in range(n_levels) if to_keep[i]],
                )
                features_df.index = new_index

        # Drop index levels that have only one unique value, for example, if the index is a MultiIndex
        # and one of the levels has only one unique value across all rows, we can drop that level
        for lvl in range(len(features_df.index.levels)):
            if len(features_df.index.get_level_values(lvl).unique()) == 1:
                features_df.index = features_df.index.droplevel(lvl)

        # Sort the columns naturally if they are strings, otherwise sort them normally
        if type(columns[0]) == str:
            sorted_columns = sorted(features_df.columns, key=natural_sort)
        else:
            sorted_columns = sorted(features_df.columns)

        # Reorder the DataFrame columns, this ensures that the columns are in a consistent order across different runs
        features_df = features_df[sorted_columns]

    # If the values in the features_dict are lists of DataFrames
    elif type(features_dict[list(features_dict.keys())[0]]) == list:

        columns = list(features_dict[list(features_dict.keys())[0]][0].columns)
        # Check if there are duplicate column names
        if len(columns) != len(set(columns)):
            raise Warning('There are duplicate column names in the single row DFs. This can cause an issue where the output '\
                        'dataframe may have an incorrect number of columns.')
        # Concatenate the DataFrames in the dictionary
        features_df = pd.concat([pd.concat(features_dict[key], keys=[key]*len(features_dict[key])) for key in features_dict.keys()])
        for lvl in range(len(features_df.index.levels)):
            if len(features_df.index.get_level_values(lvl).unique()) == 1:
                features_df.index = features_df.index.droplevel(lvl)
        if type(columns[0]) == str:
            sorted_columns = sorted(features_df.columns, key=natural_sort)
        else:
            sorted_columns = sorted(features_df.columns)
        features_df = features_df[sorted_columns]
    else:
        raise ValueError('The values in the features_dict should be either a list of dataframes or a dataframe.')

    return features_df

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

