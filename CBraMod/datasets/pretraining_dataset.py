import pickle

import lmdb
from torch.utils.data import Dataset

from utils.util import to_tensor


class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir
    ):
        super(PretrainingDataset, self).__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # self.keys = self.keys[:100000]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        patch = to_tensor(patch)
        # print(patch.shape)
        return patch



