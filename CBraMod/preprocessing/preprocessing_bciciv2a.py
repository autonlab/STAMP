'''
From https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import numpy as np
import scipy
from scipy import signal
import os
import lmdb
import pickle
from scipy.signal import butter, lfilter, resample, filtfilt

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

root_dir = '/path/to/benchmark_data/BCICIV_2a/mat'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

# files.remove('A04E.mat')
# files.remove('A04T.mat')
# files.remove('A06E.mat')
# files.remove('A06T.mat')
print(files)

files_dict = {
    'train': ['A01E.mat', 'A01T.mat', 'A02E.mat', 'A02T.mat', 'A03E.mat', 'A03T.mat',
              'A04E.mat', 'A04T.mat',
              'A05E.mat', 'A05T.mat'],
    'val': [
        'A06E.mat', 'A06T.mat',
        'A07E.mat', 'A07T.mat'
    ],
    'test': ['A08E.mat', 'A08T.mat', 'A09E.mat', 'A09T.mat'],
}



dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# for file in files:
#     if 'E' in file:
#         files_dict['train'].append(file)
#     else:
#         files_dict['test'].append(file)
#
# print(files_dict)


db = lmdb.open('/path/to/benchmark_data/BCICIV_2a/processed_inde_avg_03_50', map_size=1610612736)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        print(file)
        data = scipy.io.loadmat(os.path.join(root_dir, file))
        num = len(data['data'][0])
        # print(num)
        # print(data['data'][0, 8][0, 0][0].shape)
        # print(data['data'][0, 8][0, 0][1].shape)
        # print(data['data'][0, 8][0, 0][2].shape)
        for j in range(3, num):
            raw_data = data['data'][0, j][0, 0][0][:, :22]
            events = data['data'][0, j][0, 0][1][:, 0]
            labels = data['data'][0, j][0, 0][2][:, 0]
            length = raw_data.shape[0]
            events = events.tolist()
            events.append(length)
            # print(events)
            annos = []
            for i in range(len(events) - 1):
                annos.append((events[i], events[i + 1]))
            for i, (anno, label) in enumerate(zip(annos, labels)):
                sample = raw_data[anno[0]:anno[1]].transpose(1, 0)
                sample  = sample - np.mean(sample, axis=0, keepdims=True)
                # print(samples.shape)
                b, a = butter_bandpass(0.3, 50, 250)
                sample = lfilter(b, a, sample, -1)
                # print(sample.shape)
                sample = sample[:, 2 * 250:6 * 250]
                sample = resample(sample, 800, axis=-1)
                # print(sample.shape)
                # print(i, sample.shape, label)
                sample = sample.reshape(22, 4, 200)
                sample_key = f'{file[:-4]}-{j}-{i}'
                print(sample_key, label-1)
                data_dict = {
                    'sample': sample, 'label': label - 1
                }
                # print(label-1)
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
