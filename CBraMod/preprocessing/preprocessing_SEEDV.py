'''
Adapted from https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import os
import lmdb
import pickle
import mne
from joblib import Parallel, delayed

useless_ch = ['M1', 'M2', 'VEO', 'HEO']
trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}

root_dir = '/path/to/benchmark_data/seedv/files/EEG_raw'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
print(files)

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

def process_file(file, root_dir, trials_of_sessions, labels_of_sessions, trials_split):
    """
    Process a single EEG file and return all processed data in memory.
    """
    print(f'Processing file: {file}')
    
    # Load and preprocess the raw EEG data
    raw = mne.io.read_raw_cnt(os.path.join(root_dir, file), preload=True)
    raw.drop_channels(useless_ch)
    # raw.set_eeg_reference(ref_channels='average')
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    data_matrix = raw.get_data(units='uV')
    
    # Extract session information
    session_index = file.split('_')[1]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 200:trials_of_sessions[session_index]['end'][j] * 200]
        for j in range(15)]
    labels = labels_of_sessions[session_index]
    
    # Store all processed samples for this file
    file_data = []
    file_dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    
    # Process each trial split
    for mode in trials_split.keys():
        for index in trials_split[mode]:
            data = data_trials[index]
            label = labels[index]
            print(f'{file} - {mode} - trial {index}: {data.shape}')
            data = data.reshape(62, -1, 1, 200)
            data = data.transpose(1, 0, 2, 3)
            print(f'{file} - {mode} - trial {index} reshaped: {data.shape}')
            
            # Store each sample
            for i, sample in enumerate(data):
                sample_key = f'{file}-{index}-{i}'
                data_dict = {
                    'sample': sample, 'label': label
                }
                file_data.append((sample_key, data_dict))
                file_dataset[mode].append(sample_key)
    
    print(f'Completed processing file: {file} ({len(file_data)} samples)')
    return file_data, file_dataset

def write_to_database(all_file_data, combined_dataset, db_path):
    """
    Write all processed data to LMDB database in a single pass.
    """
    print("Writing all data to database...")
    
    # Open database once
    db = lmdb.open(db_path, map_size=15614542346)
    
    # Write all data in a single transaction for maximum efficiency
    txn = db.begin(write=True)
    
    total_samples = 0
    for file_data, _ in all_file_data:
        for sample_key, data_dict in file_data:
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            total_samples += 1
    
    # Store the dataset keys
    txn.put(key='__keys__'.encode(), value=pickle.dumps(combined_dataset))
    
    # Commit all writes at once
    txn.commit()
    db.close()
    
    print(f"Successfully wrote {total_samples} samples to database")

def main():
    """
    Main function to orchestrate parallel processing and sequential writing.
    """
    db_path = '/path/to/benchmark_data/seedv/processed'
    
    # Step 1: Process files in parallel (CPU-intensive operations)
    print(f"Processing {len(files)} files in parallel...")
    results = Parallel(n_jobs=40, verbose=10)(
        delayed(process_file)(
            file, root_dir, trials_of_sessions, labels_of_sessions, trials_split
        ) for file in files
    )
    
    # Step 2: Combine results from all files
    combined_dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    
    for file_data, file_dataset in results:
        for mode in ['train', 'val', 'test']:
            combined_dataset[mode].extend(file_dataset[mode])
    
    # Step 3: Write everything to database sequentially (I/O operations)
    write_to_database(results, combined_dataset, db_path)
    
    print("Processing complete!")
    print(f"Train samples: {len(combined_dataset['train'])}")
    print(f"Val samples: {len(combined_dataset['val'])}")
    print(f"Test samples: {len(combined_dataset['test'])}")

if __name__ == "__main__":
    main()