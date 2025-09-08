import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb

'''Adapted from https://github.com/wjq-learning/CBraMod'''

selected_channels = {
    '01_tcp_ar': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ],
    '02_tcp_le': [
            'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
            'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
            'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
    ],
    '03_tcp_ar_a': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ]
}

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


#遍历文件夹
def iter_files(rootDir):
    #遍历根目录
    file_path_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            # print(file_name)
            file_path_list.append(file_name)
    return file_path_list

def preprocessing_recording(file_path, file_key_list: list, db: lmdb.open):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    if '02_tcp_le' in file_path:
        for ch in selected_channels['02_tcp_le']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['02_tcp_le'], ordered=True)
    elif '01_tcp_ar' in file_path:
        for ch in selected_channels['01_tcp_ar']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['01_tcp_ar'], ordered=True)
    elif '03_tcp_ar_a' in file_path:
        for ch in selected_channels['03_tcp_ar_a']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['03_tcp_ar_a'], ordered=True)
    else:
        return
    # print(raw.info)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))
    eeg_array = raw.to_data_frame().values
    # print(raw.info)
    eeg_array = eeg_array[:, 1:]
    points, chs = eeg_array.shape
    if points < 300 * 200:
        return
    a = points % (30 * 200)
    eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
    # print(eeg_array.shape)
    eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    eeg_array = eeg_array.transpose(0, 3, 1, 2)
    print(eeg_array.shape)
    file_name = file_path.split('/')[-1][:-4]

    for i, sample in enumerate(eeg_array):
        # print(i, sample.shape)
        if np.max(np.abs(sample)) < 100:
            sample_key = f'{file_name}_{i}'
            print(sample_key)
            file_key_list.append(sample_key)
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
            txn.commit()

if __name__ == '__main__':
    setup_seed(1)
    file_path_list = iter_files('path...')

    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    # print(file_path_list)
    db = lmdb.open(r'path...', map_size=1649267441664)
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, db)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()
