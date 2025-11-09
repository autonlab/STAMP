'''
Adapted from https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import sys
sys.path.append('/path/to/STAMP/CBraMod/preprocessing')

import pickle
import os
import numpy as np
from joblib import Parallel, delayed
from stamp.datasets.lmdb_writer import LMDBWriter  # Assuming this is your LMDB writer class
import logging
import json
from scipy import signal as scipy_signal

'''Adapted from https://github.com/wjq-learning/CBraMod'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# dump chb23 and chb24 to test, ch21 and ch22 to val, and the rest to train
test_pats = ["chb23", "chb24"]
val_pats = ["chb21", "chb22"]
train_pats = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
]
channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
SAMPLING_RATE = 256

def process_single_patient(params):
    """Process a single patient's recordings and return segments"""
    folder, patient_split = params

    try:
        logger.info(f"Processing patient {folder}")
        patient_path = os.path.join(root, folder)

        if not os.path.isdir(patient_path):
            logger.warning(f"Patient directory not found: {patient_path}")
            return [], []

        samples = []
        error_files = []

        # Process each recording file for this patient
        for f in os.listdir(patient_path):
            if not f.endswith('.pkl'):
                continue

            try:
                logger.info(f"Processing {folder}/{f}")
                record = pickle.load(open(os.path.join(patient_path, f), "rb"))

                # Extract signal data for specified channels
                signal = []
                for channel in channels:
                    if channel in record:
                        signal.append(record[channel])
                    else:
                        raise ValueError(f"Channel {channel} not found in record {f}")

                signal = np.array(signal)  # Shape: (16, time_points)

                # Get seizure times
                if "times" in record["metadata"]:
                    seizure_times = record["metadata"]["times"]
                else:
                    seizure_times = []

                # Split signal into 10-second segments
                for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
                    segment = signal[:, i : i + 10 * SAMPLING_RATE]
                    if segment.shape[1] == 10 * SAMPLING_RATE:
                        # Determine if segment contains seizures
                        label = 0
                        for seizure_time in seizure_times:
                            if (i < seizure_time[0] < i + 10 * SAMPLING_RATE or
                                i < seizure_time[1] < i + 10 * SAMPLING_RATE):
                                label = 1
                                break

                        segment = scipy_signal.resample(segment, 2000, axis=1)
                        segment = segment.reshape(16,10,200)
                        # Create filename for this segment
                        filename = f"{f.split('.')[0]}-{i}"
                        samples.append((segment, label, filename))

                # Add additional seizure segments (overlapping)
                for idx, seizure_time in enumerate(seizure_times):
                    for i in range(
                        max(0, seizure_time[0] - SAMPLING_RATE),
                        min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                        5 * SAMPLING_RATE,
                    ):
                        segment = signal[:, i : i + 10 * SAMPLING_RATE]
                        label = 1

                        segment = scipy_signal.resample(segment, 2000, axis=1)
                        segment = segment.reshape(16,10,200)

                        filename = f"{f.split('.')[0]}-s-{idx}-add-{i}"
                        samples.append((segment, label, filename))

            except Exception as e:
                logger.error(f"Error processing file {folder}/{f}: {str(e)}")
                error_files.append(f"{folder}/{f}")
                continue

        logger.info(f"Patient {folder}: processed {len(samples)} segments, {len(error_files)} errors")
        return samples, error_files

    except Exception as e:
        logger.error(f"Error processing patient {folder}: {str(e)}")
        return [], [f"Patient {folder}: {str(e)}"]

def process_patients_for_split(patient_params, lmdb_path, split_name, map_size, chunk_size=5, n_jobs=-1):
    """Process patients in chunks and write to LMDB for a specific split"""
    logger.info(f"Processing {split_name} split with {len(patient_params)} patients")

    # Split patients into chunks
    patient_chunks = [patient_params[i:i + chunk_size] for i in range(0, len(patient_params), chunk_size)]

    with LMDBWriter(lmdb_path, map_size=map_size) as writer:
        all_error_files = []

        for chunk_idx, chunk in enumerate(patient_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(patient_chunks)} for {split_name} ({len(chunk)} patients)")

            # Process chunk in parallel using joblib
            chunk_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_single_patient)(params)
                for params in chunk
            )

            logger.info(f'Writing chunk to lmdb...')
            # Write chunk results to LMDB
            chunk_samples = 0
            for samples, error_files in chunk_results:
                all_error_files.extend(error_files)
                for segment_data, segment_label, filename in samples:
                    # Store individual segment with its label
                    writer.write_sample(segment_data, segment_label, filename, dtype=np.float64)
                    chunk_samples += 1

            logger.info(f"Completed chunk {chunk_idx + 1} for {split_name}: wrote {chunk_samples} segments")

            # Clear chunk data from memory
            del chunk_results

        # Log errors
        if all_error_files:
            error_file_path = f"chbmit-process-error-files-{split_name}.txt"
            with open(error_file_path, "w") as f:
                for error_file in all_error_files:
                    f.write(error_file + "\n")
            logger.warning(f"Wrote {len(all_error_files)} error files to {error_file_path}")

        logger.info(f"Completed {split_name} split: total segments = {writer.get_count()}")

if __name__ == "__main__":
    """Main processing function"""

    root = "/path/to/benchmark_data/chbmit/processed"
    output_dir = "/path/to/benchmark_data/chbmit/processed_lmdb"
    chunk_size = 5
    n_jobs = -1
    # Create LMDB output directory
    lmdb_output_dir = os.path.join(output_dir)
    os.makedirs(lmdb_output_dir, exist_ok=False)

    # Get all patient folders
    all_folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

    # Prepare parameters for each split
    train_params = []
    val_params = []
    test_params = []

    for folder in all_folders:
        if folder in test_pats:
            test_params.append((folder, "test"))
        elif folder in val_pats:
            val_params.append((folder, "val"))
        elif folder in train_pats:
            train_params.append((folder, "train"))
        else:
            logger.warning(f"Unknown patient {folder}")

    logger.info(f"Split: Train={len(train_params)}, Val={len(val_params)}, Test={len(test_params)} patients")

    # Create separate LMDB databases for each split
    train_lmdb_path = os.path.join(lmdb_output_dir, "train")
    val_lmdb_path = os.path.join(lmdb_output_dir, "val")
    test_lmdb_path = os.path.join(lmdb_output_dir, "test")

    logger.info("Starting CHB-MIT dataset processing...")

    # Process each split separately
    logger.info("Processing training patients...")
    process_patients_for_split(train_params, train_lmdb_path, "train", map_size=70 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing validation patients...")
    process_patients_for_split(val_params, val_lmdb_path, "val", map_size=20 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing test patients...")
    process_patients_for_split(test_params, test_lmdb_path, "test", map_size=20 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)

    logger.info("Processing complete!")
