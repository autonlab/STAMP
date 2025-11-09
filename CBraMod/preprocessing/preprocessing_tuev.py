'''
Adapted from https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import mne
import numpy as np
import os
from tqdm import tqdm
import logging
from pathlib import Path
from joblib import Parallel, delayed
from stamp.datasets.lmdb_writer import LMDBWriter

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]

def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]], # 0
            signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]], # 1
            signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]], # 2
            signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]], # 3
            signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]], # 4
            signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]], # 5
            signals[signal_names["EEG T4-REF"]]- signals[signal_names["EEG T6-REF"]], # 6
            signals[signal_names["EEG T6-REF"]]- signals[signal_names["EEG O2-REF"]], # 7
            signals[signal_names["EEG FP1-REF"]]- signals[signal_names["EEG F3-REF"]], # 14
            signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]], # 15
            signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]], # 16
            signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]], # 17
            signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]], # 18
            signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]], # 19
            signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]], # 20
            signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]], # 21
        )
    )
    return new_signals

def readEDF(fileName):
    try:
        Rawdata = mne.io.read_raw_edf(fileName, preload=True, verbose=False)
        Rawdata.resample(200)
        Rawdata.filter(l_freq=0.3, h_freq=75, verbose=False)
        Rawdata.notch_filter(60, verbose=False)

        _, times = Rawdata[:]
        signals = Rawdata.get_data(units='uV')
        RecFile = fileName[0:-3] + "rec"
        eventData = np.genfromtxt(RecFile, delimiter=",")
        Rawdata.close()

        return [signals, times, eventData, Rawdata]
    except Exception as e:
        logger.error(f"Error processing {fileName}: {str(e)}")
        return None

def process_single_file(file_path):
    try:
        result = readEDF(file_path)
        if result is None:
            return 0

        signals, times, event, Rawdata = result
        signals = convert_signals(signals, Rawdata)
        signals, offending_channels, labels = BuildEvents(signals, times, event)

        filename_base = Path(file_path).stem

        samples = []
        for idx, (signal, offending_channel, label) in enumerate(
            zip(signals, offending_channels, labels)
        ):
            signal = signal.reshape(16, 5, 200)
            samples.append((signal, int(label[0]), filename_base))

        print(f"Processed {file_path}: {len(samples)} samples")
        return samples

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

def get_subject_id(filename):
    """Extract subject ID from filename"""
    return filename.split("_")[0]

def collect_edf_files(base_dir):
    """Collect all EDF files and group by subject"""
    edf_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    return edf_files

def split_subjects(subjects, train_ratio=0.8):
    """Split subjects into train and validation sets"""
    subjects = sorted(subjects) # Ensures splits are always the same
    split_idx = int(len(subjects) * train_ratio)
    return subjects[:split_idx], subjects[split_idx:]

def process_files_parallel(file_list, lmdb_path, map_size, chunk_size = 400, n_jobs=-1):
    """Process files in parallel using joblib"""

    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    logger.info(f"Processing {len(file_list)} files in {len(file_chunks)} chunks of size {chunk_size}")

    with LMDBWriter(lmdb_path, map_size=map_size) as writer:
        total_samples = 0
        for chunk_idx, file_chunk in enumerate(file_chunks):
            chunk_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_single_file)(file_path)
                for file_path in file_chunk
            )

            chunk_samples = []
            for file_samples in chunk_results:
                chunk_samples.extend(file_samples)

            # Write chunk to LMDB
            logger.info(f"Writing {len(chunk_samples)} samples from chunk {chunk_idx + 1}")
            for signal, label, filename in tqdm(chunk_samples, desc=f"Writing chunk {chunk_idx + 1}"):
                writer.write_sample(signal, label, filename, dtype=np.float64)

            total_samples += len(chunk_samples)

            # Clear chunk data from memory
            del chunk_samples
            del chunk_results

            logger.info(f"Completed chunk {chunk_idx + 1}, total samples so far: {total_samples}")

    logger.info(f"Processed {len(file_list)} files, generated {total_samples} samples")
    return total_samples

def main():
    """Main processing pipeline"""
    # Configuration
    root = "/path/to/benchmark_data/tuev/v2.0.1/edf"
    output_dir = "/path/to/benchmark_data/tuev"
    n_jobs = -1
    chunk_size = 400

    # Create output directories
    train_dump_folder = os.path.join(output_dir, "processed", "train")
    val_dump_folder = os.path.join(output_dir, "processed", "val")
    test_dump_folder = os.path.join(output_dir, "processed", "test")

    for folder in [train_dump_folder, val_dump_folder, test_dump_folder]:
        os.makedirs(folder, exist_ok=True)

    # Process training data with subject-based splitting
    logger.info("Processing training data...")
    train_base_dir = os.path.join(root, "train")
    train_files = collect_edf_files(train_base_dir)

    test_base_dir = os.path.join(root, "eval")
    test_files = collect_edf_files(test_base_dir)

    # Group files by subject
    subject_files = {}
    for file_path in train_files:
        filename = Path(file_path).name
        subject_id = get_subject_id(filename)
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(file_path)

    # Split subjects
    subjects = list(subject_files.keys())
    train_subjects, val_subjects = split_subjects(subjects, train_ratio=0.8)

    logger.info(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    logger.info(f"Validation subjects ({len(val_subjects)}): {val_subjects}")

    # Collect files for each split
    train_file_list = []
    val_file_list = []

    for subject in train_subjects:
        train_file_list.extend(subject_files[subject])
    for subject in val_subjects:
        val_file_list.extend(subject_files[subject])

    # Process train files
    logger.info("Processing training files...")
    process_files_parallel(train_file_list, train_dump_folder, map_size=13 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)

    # Process validation files
    logger.info("Processing validation files...")
    process_files_parallel(val_file_list, val_dump_folder, map_size=7 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)

    # Process test data (eval directory)
    logger.info("Processing test data...")
    process_files_parallel(test_files, test_dump_folder, map_size=7 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)

    logger.info("Processing complete!")

if __name__ == "__main__":
    # Suppress MNE warnings for cleaner output
    mne.set_log_level('ERROR')
    main()


