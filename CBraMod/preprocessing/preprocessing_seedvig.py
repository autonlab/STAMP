'''
From https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import h5py
import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import pandas as pd

data_dir = '/path/to/benchmark_data/seedvig/files/Raw_Data'
labels_dir = '/path/to/benchmark_data/seedvig/files/perclos_labels'

files = [file for file in os.listdir(data_dir)]
files = sorted(files)

files_dict = {
    'train': files[:15],
    'val': files[15:19],
    'test': files[19:23],
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open('/path/to/benchmark_data/seedvig/processed', map_size=6000000000)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        eeg = scipy.io.loadmat(os.path.join(data_dir, file))['EEG'][0][0][0]
        labels = scipy.io.loadmat(os.path.join(labels_dir, file))['perclos']
        print(eeg.shape, labels.shape)
        eeg = eeg.reshape(885, 8, 200, 17)
        eeg = eeg.transpose(0, 3, 1, 2)
        labels = labels[:, 0]
        print(eeg.shape, labels.shape)
        for i, (sample, label) in enumerate(zip(eeg, labels)):
            sample_key = f'{file[:-4]}-{i}'
            print(sample_key)
            data_dict = {
                'sample': sample, 'label': label
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()