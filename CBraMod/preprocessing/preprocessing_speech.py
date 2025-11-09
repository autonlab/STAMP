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

train_dir = '/path/to/benchmark_data/speech/files/Training set'
val_dir = '/path/to/benchmark_data/speech/files/Validation set'
test_dir = '/path/to/benchmark_data/speech/files/Test set'



files_dict = {
    'train':sorted([file for file in os.listdir(train_dir)]),
    'val':sorted([file for file in os.listdir(val_dir)]),
    'test':sorted([file for file in os.listdir(test_dir)]),
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open('/path/to/benchmark_data/speech/processed', map_size=3000000000)

for file in files_dict['train']:
    data = scipy.io.loadmat(os.path.join(train_dir, file))
    print(data['epo_train'][0][0][0])
    eeg = data['epo_train'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_train'][0][0][5].transpose(1, 0)
    eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = signal.resample(eeg, 600, axis=2).reshape(300, 64, 3, 200)
    print(eeg.shape, labels.shape)
    for i, (sample, label) in enumerate(zip(eeg, labels)):
        sample_key = f'train-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['train'].append(sample_key)


for file in files_dict['val']:
    data = scipy.io.loadmat(os.path.join(val_dir, file))
    eeg = data['epo_validation'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_validation'][0][0][5].transpose(1, 0)
    eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = signal.resample(eeg, 600, axis=2).reshape(50, 64, 3, 200)
    print(eeg.shape, labels.shape)
    for i, (sample, label) in enumerate(zip(eeg, labels)):
        sample_key = f'val-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['val'].append(sample_key)


df = pd.read_excel("/path/to/benchmark_data/speech/files/Track3_Answer Sheet_Test.xlsx")
df_=df.head(53)
all_labels=df_.values
print(all_labels.shape)
all_labels = all_labels[2:, 1:][:, 1:30:2].transpose(1, 0)
print(all_labels.shape)
print(all_labels)

for j, file in enumerate(files_dict['test']):
    data = h5py.File(os.path.join(test_dir, file))
    eeg = data['epo_test']['x'][:]
    labels = all_labels[j]
    eeg = eeg[:, :, -768:]
    eeg = signal.resample(eeg, 600, axis=2).reshape(50, 64, 3, 200)
    print(eeg.shape, labels.shape)
    for i, (sample, label) in enumerate(zip(eeg, labels)):
        sample_key = f'test-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label-1,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['test'].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()