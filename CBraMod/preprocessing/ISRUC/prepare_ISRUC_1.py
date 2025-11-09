'''
Adapted from https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

from edf_ import read_raw_edf
import mne
import os
import numpy as np
from joblib import Parallel, delayed
import logging
from stamp.datasets.lmdb_writer import LMDBWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_subject(params):
    """Process a single subject's PSG and label files"""
    psg_f_name, label_f_name, subject_id, label2id, dir_path = params

    try:
        logger.info(f"Processing subject {subject_id}")

        # Read and process PSG data
        raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True, verbose=False)
        raw.filter(0.3, 35, fir_design='firwin', verbose=False)
        raw.notch_filter((50), verbose=False)

        psg_array = raw.to_data_frame().values
        psg_array = psg_array[:, 1:]  # Remove first column
        psg_array = psg_array[:, 2:8]  # Select channels 2-7 (6 channels)

        # Reshape to 30-second epochs at 200Hz
        i = psg_array.shape[0] % (30 * 200)
        if i > 0:
            psg_array = psg_array[:-i, :]
        psg_array = psg_array.reshape(-1, 30 * 200, 6)

        # Group into sequences of 20 epochs
        a = psg_array.shape[0] % 20
        if a > 0:
            psg_array = psg_array[:-a, :, :]
        psg_array = psg_array.reshape(-1, 20, 30 * 200, 6)
        epochs_seq = psg_array.transpose(0, 1, 3, 2)

        # Process labels
        labels_list = []
        for line in open(os.path.join(dir_path, label_f_name)).readlines():
            line_str = line.strip()
            if line_str != '':
                labels_list.append(label2id[line_str])

        labels_array = np.array(labels_list)
        if a > 0:
            labels_array = labels_array[:-a]
        labels_seq = labels_array.reshape(-1, 20)  # Shape: (n_sequences, 20)

        # Create samples list
        samples = []
        for seq_idx, (seq_data, seq_labels) in enumerate(zip(epochs_seq, labels_seq)):
            for epoch_idx, (epoch_data, epoch_label) in enumerate(zip(seq_data, seq_labels)):
                # Create filename for this individual epoch
                filename = f'ISRUC-group1-{subject_id}-seq{seq_idx}-epoch{epoch_idx}'

                # epoch_data shape: (6, 6000) - 6 channels, 6000 time points
                # Store individual epoch with its label
                samples.append((epoch_data, int(epoch_label), filename))

        logger.info(f"Subject {subject_id}: processed {len(samples)} individual epochs")
        raw.close()
        return samples, []

    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {str(e)}")
        return [], [f"Subject {subject_id}: {str(e)}"]

def process_split_chunked(subject_params, lmdb_path, map_size, chunk_size=10, n_jobs=-1):
    """Process subjects in chunks and write to LMDB"""
    logger.info(f"Processing {len(subject_params)} subjects")

    # Split subjects into chunks
    subject_chunks = [subject_params[i:i + chunk_size] for i in range(0, len(subject_params), chunk_size)]

    with LMDBWriter(lmdb_path, map_size=map_size) as writer:  # 30GB map size
        all_error_files = []

        for chunk_idx, chunk in enumerate(subject_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(subject_chunks)} ({len(chunk)} subjects)")

            # Process chunk in parallel using joblib
            chunk_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_single_subject)(params)
                for params in chunk
            )

            # Write chunk results to LMDB
            chunk_samples = 0
            for samples, error_files in chunk_results:
                all_error_files.extend(error_files)
                for epoch_data, epoch_label, filename in samples:
                    # Store individual epoch with its label (like TUAB)
                    writer.write_sample(epoch_data, epoch_label, filename, dtype=np.float64)
                    chunk_samples += 1

            logger.info(f"Completed chunk {chunk_idx + 1}: wrote {chunk_samples} items")

            # Clear chunk data from memory
            del chunk_results

        # Log errors
        if all_error_files:
            error_file_path = "isruc-process-error-files.txt"
            with open(error_file_path, "w") as f:
                for error_file in all_error_files:
                    f.write(error_file + "\n")
            logger.warning(f"Wrote {len(all_error_files)} error files to {error_file_path}")

        logger.info(f"Completed processing: total items = {writer.get_count()}")

if __name__ == "__main__":
    """Main processing function"""
    # Suppress MNE warnings for cleaner output
    mne.set_log_level('ERROR')

    dir_path = r'/path/to/benchmark_data/isruc/files/extracted'
    output_dir = r'/path/to/benchmark_data/isruc'
    n_jobs = -1
    chunk_size = 10

    lmdb_output_dir = os.path.join(output_dir, "processed")
    os.makedirs(lmdb_output_dir, exist_ok=False)

    psg_f_names = []
    label_f_names = []
    for i in range(1, 101):
        numstr = str(i)
        psg_f_names.append(f'{dir_path}/{numstr}/{numstr}.rec')
        label_f_names.append(f'{dir_path}/{numstr}/{numstr}_1.txt')

    psg_label_f_pairs = []
    for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
        if psg_f_name[:-4] == label_f_name[:-6]:
            psg_label_f_pairs.append((psg_f_name, label_f_name))

    logger.info(f"Found {len(psg_label_f_pairs)} valid file pairs")

    label2id = {'0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '5': 4,}
    logger.info(f"Label mapping: {label2id}")

    # Prepare parameters for processing and split by subject ID
    train_params = []
    val_params = []
    test_params = []

    # Prepare parameters for processing
    subject_params = []
    for subject_id, (psg_f_name, label_f_name) in enumerate(psg_label_f_pairs, 1):
        params = (psg_f_name, label_f_name, subject_id, label2id, dir_path)

        if subject_id < 80:  # subjects 1-80 for training
            train_params.append(params)
        elif subject_id < 90:  # subjects 81-90 for validation
            val_params.append(params)
        else:  # subjects 91-100 for testing
            test_params.append(params)

    logger.info(f"Split: Train={len(train_params)}, Val={len(val_params)}, Test={len(test_params)} subjects")

    # Process all subjects
    train_dump_folder = os.path.join(output_dir, "processed", "train")
    val_dump_folder = os.path.join(output_dir, "processed", "val")
    test_dump_folder = os.path.join(output_dir, "processed", "test")

    logger.info("Starting ISRUC dataset processing...")

    logger.info("Processing training files...")
    process_split_chunked(train_params, train_dump_folder, map_size=20 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing validation files...")
    process_split_chunked(val_params, val_dump_folder, map_size=5 * 1024**3,chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing test data...")
    process_split_chunked(test_params, test_dump_folder, map_size=5 * 1024**3,chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing complete!")

    logger.info("Processing complete!")



