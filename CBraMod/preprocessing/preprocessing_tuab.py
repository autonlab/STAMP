import os
from joblib import Parallel, delayed
import mne
import logging
import numpy as np
from lmdb_writer import LMDBWriter

'''Adapted from https://github.com/wjq-learning/CBraMod'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# we need these channels
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F7-REF']],  # 0
# (signals[signal_names['EEG F7-REF']] - signals[signal_names['EEG T3-REF']]),  # 1
# (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG T5-REF']]),  # 2
# (signals[signal_names['EEG T5-REF']] - signals[signal_names['EEG O1-REF']]),  # 3
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F8-REF']]),  # 4
# (signals[signal_names['EEG F8-REF']] - signals[signal_names['EEG T4-REF']]),  # 5
# (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG T6-REF']]),  # 6
# (signals[signal_names['EEG T6-REF']] - signals[signal_names['EEG O2-REF']]),  # 7
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F3-REF']]),  # 14
# (signals[signal_names['EEG F3-REF']] - signals[signal_names['EEG C3-REF']]),  # 15
# (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG P3-REF']]),  # 16
# (signals[signal_names['EEG P3-REF']] - signals[signal_names['EEG O1-REF']]),  # 17
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F4-REF']]),  # 18
# (signals[signal_names['EEG F4-REF']] - signals[signal_names['EEG C4-REF']]),  # 19
# (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG P4-REF']]),  # 20
# (signals[signal_names['EEG P4-REF']] - signals[signal_names['EEG O2-REF']]))) # 21
standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def process_subject_files(params):
    fetch_folder, sub, label = params
    samples = []
    error_files = []
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True)
                raw.resample(200)
                raw.filter(l_freq=0.3, h_freq=75)
                raw.notch_filter((60))
                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()[:16]
                try:
                    channeled_data[0] = (
                        raw_data[ch_name.index("EEG FP1-REF")]
                        - raw_data[ch_name.index("EEG F7-REF")]
                    )
                    channeled_data[1] = (
                        raw_data[ch_name.index("EEG F7-REF")]
                        - raw_data[ch_name.index("EEG T3-REF")]
                    )
                    channeled_data[2] = (
                        raw_data[ch_name.index("EEG T3-REF")]
                        - raw_data[ch_name.index("EEG T5-REF")]
                    )
                    channeled_data[3] = (
                        raw_data[ch_name.index("EEG T5-REF")]
                        - raw_data[ch_name.index("EEG O1-REF")]
                    )
                    channeled_data[4] = (
                        raw_data[ch_name.index("EEG FP2-REF")]
                        - raw_data[ch_name.index("EEG F8-REF")]
                    )
                    channeled_data[5] = (
                        raw_data[ch_name.index("EEG F8-REF")]
                        - raw_data[ch_name.index("EEG T4-REF")]
                    )
                    channeled_data[6] = (
                        raw_data[ch_name.index("EEG T4-REF")]
                        - raw_data[ch_name.index("EEG T6-REF")]
                    )
                    channeled_data[7] = (
                        raw_data[ch_name.index("EEG T6-REF")]
                        - raw_data[ch_name.index("EEG O2-REF")]
                    )
                    channeled_data[8] = (
                        raw_data[ch_name.index("EEG FP1-REF")]
                        - raw_data[ch_name.index("EEG F3-REF")]
                    )
                    channeled_data[9] = (
                        raw_data[ch_name.index("EEG F3-REF")]
                        - raw_data[ch_name.index("EEG C3-REF")]
                    )
                    channeled_data[10] = (
                        raw_data[ch_name.index("EEG C3-REF")]
                        - raw_data[ch_name.index("EEG P3-REF")]
                    )
                    channeled_data[11] = (
                        raw_data[ch_name.index("EEG P3-REF")]
                        - raw_data[ch_name.index("EEG O1-REF")]
                    )
                    channeled_data[12] = (
                        raw_data[ch_name.index("EEG FP2-REF")]
                        - raw_data[ch_name.index("EEG F4-REF")]
                    )
                    channeled_data[13] = (
                        raw_data[ch_name.index("EEG F4-REF")]
                        - raw_data[ch_name.index("EEG C4-REF")]
                    )
                    channeled_data[14] = (
                        raw_data[ch_name.index("EEG C4-REF")]
                        - raw_data[ch_name.index("EEG P4-REF")]
                    )
                    channeled_data[15] = (
                        raw_data[ch_name.index("EEG P4-REF")]
                        - raw_data[ch_name.index("EEG O2-REF")]
                    )
                except:
                    with open("tuab-process-error-files.txt", "a") as f:
                        f.write(file + "\n")
                    continue

                for i in range(channeled_data.shape[1] // 2000):
                    base_filename = file.split(".")[0] + "_" + str(i) + f'_y{label}' # We store the label in the file name
                    segment = channeled_data[:, i * 2000 : (i + 1) * 2000]
                    segment_reshaped = segment.reshape(16, 10, 200)
                    samples.append((segment_reshaped, label, base_filename))

                raw.close()
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                error_files.append(file)
                continue

    logger.info(f"Processed subject {sub}: {len(samples)} samples, {len(error_files)} errors")
    return samples, error_files

def process_split_chunked(subject_params, lmdb_path, split_name, map_size, chunk_size=10, n_jobs=-1):
    """Process a split (train/val/test) in chunks to manage memory"""
    logger.info(f"Processing {split_name} split with {len(subject_params)} subjects")

    # Split subjects into chunks
    subject_chunks = [subject_params[i:i + chunk_size] for i in range(0, len(subject_params), chunk_size)]

    with LMDBWriter(lmdb_path, map_size=map_size) as writer:
        all_error_files = []

        for chunk_idx, chunk in enumerate(subject_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(subject_chunks)} for {split_name} ({len(chunk)} subjects)")

            # Process chunk in parallel using joblib
            chunk_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_subject_files)(params)
                for params in chunk
            )

            # Write chunk results to LMDB
            chunk_samples = 0
            for samples, error_files in chunk_results:
                all_error_files.extend(error_files)
                for signal, label, filename in samples:
                    writer.write_sample(signal, label, filename, dtype=np.float64)
                    chunk_samples += 1

            logger.info(f"Completed chunk {chunk_idx + 1} for {split_name}: wrote {chunk_samples} samples")

            # Clear chunk data from memory
            del chunk_results

        # Log errors
        if all_error_files:
            error_file_path = f"tuab-process-error-files-{split_name}.txt"
            with open(error_file_path, "w") as f:
                for error_file in all_error_files:
                    f.write(error_file + "\n")
            logger.warning(f"Wrote {len(all_error_files)} error files to {error_file_path}")

        logger.info(f"Completed {split_name} split: total samples = {writer.get_count()}")

if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    # root to abnormal dataset
    root = "/path/to/benchmark_data/tuab/v3.0.1/edf"
    output_dir = '/path/to/benchmark_data/tuab'
    channel_std = "01_tcp_ar"
    n_jobs = -1
    chunk_size = 800

    # seed = 4523
    # np.random.seed(seed)
    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    train_val_a_sub.sort(key=lambda x: x)

    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )
    print('train_a_sub:', train_a_sub)
    print('val_a_sub:', val_a_sub)

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    train_val_n_sub.sort(key=lambda x: x)

    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )
    print('train_n_sub:', train_n_sub)
    print('val_n_sub:', val_n_sub)

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(output_dir, "processed")):
        os.makedirs(os.path.join(output_dir, "processed"))

    train_dump_folder = os.path.join(output_dir, "processed", "train")
    val_dump_folder = os.path.join(output_dir, "processed", "val")
    test_dump_folder = os.path.join(output_dir, "processed", "test")

    for folder in [train_dump_folder, val_dump_folder, test_dump_folder]:
        os.makedirs(folder, exist_ok=True)

    train_params = []
    for train_sub in train_a_sub:
        train_params.append([train_val_abnormal, train_sub, 1])
    for train_sub in train_n_sub:
        train_params.append([train_val_normal, train_sub, 0])

    val_params = []
    for val_sub in val_a_sub:
        val_params.append([train_val_abnormal, val_sub, 1])
    for val_sub in val_n_sub:
        val_params.append([train_val_normal, val_sub, 0])

    test_params = []
    for test_sub in test_a_sub:
        test_params.append([test_abnormal, test_sub, 1])
    for test_sub in test_n_sub:
        test_params.append([test_normal, test_sub, 0])

    logger.info("Starting TUAB dataset processing...")

    logger.info("Processing training files...")
    process_split_chunked(train_params, train_dump_folder, "train", map_size = 85 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing validation files...")
    process_split_chunked(val_params, val_dump_folder, "val", map_size = 20 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing test data...")
    process_split_chunked(test_params, test_dump_folder, "test", map_size = 20 * 1024**3, chunk_size=chunk_size, n_jobs=n_jobs)
    logger.info("Processing complete!")

    print('Done!')