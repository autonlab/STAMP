import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
from functools import partial
import json
from CBraMod.utils.util import to_tensor
from stamp.datasets.utils import get_dataset_params

'''
This class should work with the processed LMDBs for TUAB, TUEV, ISRUC, and CHB-MIT
'''

class CustomLMDBNumpyDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            data_dir,
            mode,
            n_temporal_channels,
            n_spatial_channels,
            orig_seq_len,
            tdr,
            seed,
            pad_to_len=0,
            reshape_data=False,
            check_reshaped_data=False
    ):
        super(CustomLMDBNumpyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.pad_to_len = pad_to_len
        self.reshape_data = reshape_data
        self.check_reshaped_data = check_reshaped_data
        self.mode = mode
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.orig_seq_len = orig_seq_len
        self.tdr = tdr if tdr is not None else 1.0
        self.seed = seed

        # Initialize LMDB connection
        self._init_db()

    def _init_db(self):
        """Initialize LMDB database connection"""
        self.db = lmdb.open(self.data_dir + f'/{self.mode}', readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            keys_bytes = txn.get(b'__keys__')
            keys_str = json.loads(keys_bytes.decode())
            self.keys = [k.encode() for k in keys_str]  # Convert back to bytes
            if self.tdr < 1.0 and self.mode == 'train':
                print(f"Using training data ratio of {self.tdr}")
                length = len(self.keys)
                # Shuffle keys
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.keys)
                self.keys = self.keys[:int(length * self.tdr)]
            self.length = len(self.keys)
            print(f"Loaded {self.length} keys from stored __keys__")

    def __getstate__(self):
        """Prepare object for pickling - remove unpicklable LMDB connection"""
        state = self.__dict__.copy()
        # Remove the LMDB database connection
        if 'db' in state:
            del state['db']
        return state

    def __setstate__(self, state):
        """Restore object after unpickling - recreate LMDB connection"""
        self.__dict__.update(state)
        # Recreate the LMDB database connection
        self._init_db()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range")

        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            key = self.keys[idx]
            data_bytes = txn.get(key)

            if data_bytes is None:
                raise IndexError(f"Sample {idx} not found")

            # Extract label from key
            key_str = key.decode()
            parts = key_str.split('_')
            label = int(parts[-1][1:])  # Remove 'y' prefix

            # Reconstruct signal
            sample = np.frombuffer(data_bytes, dtype=np.float64) # Shape: (n_spatial_channels * n_temporal_channels * orig_seq_len,)
            sample = sample.reshape(self.n_spatial_channels, self.n_temporal_channels, self.orig_seq_len) # Shape: (n_spatial_channels, n_temporal_channels, orig_seq_len)

        return sample/100, label, key

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch]) # Shape: (batch_size, n_spatial_channels, n_temporal_channels, orig_seq_len)
        y_label = np.array([x[1] for x in batch]) # Shape: (batch_size,)
        sample_keys = [x[2] for x in batch] # List of sample keys

        # Handle padding
        if self.pad_to_len and x_data.shape[-1] < self.pad_to_len:
            pad_width = self.pad_to_len - x_data.shape[-1]
            x_data = np.pad(x_data, pad_width=((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant') # Pad to (batch_size, n_spatial_channels, n_temporal_channels, pad_to_len)

        if self.reshape_data:
            batch_size = x_data.shape[0]
            
            x_data = x_data.reshape(-1, x_data.shape[-1]) # Reshape to (batch_size * n_spatial_channels * n_temporal_channels, seq_len)
            x_data = np.expand_dims(x_data, axis=1) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, 1, seq_len)

            batch_indices = np.repeat(np.arange(batch_size), self.n_spatial_channels * self.n_temporal_channels)

            sample_keys = np.array(sample_keys, dtype=object)[batch_indices].tolist() # Shape: (batch_size * n_spatial_channels * n_temporal_channels)

        return to_tensor(x_data), to_tensor(y_label).long(), sample_keys

    def collate_with_mask(dataset, batch, orig_seq_len):
        x_data, y_label, sample_keys = dataset.collate(batch)
        # X_data shape: (batch_size, 1, seq_len)
        # y_label shape: (batch_size,)

        # Create a mask for the sequence length
        mask = torch.ones(x_data.shape[0], x_data.shape[-1], dtype=torch.bool)  # Shape: (batch_size, seq_len)
        # Zero out the padding part of the mask
        pad_width = dataset.pad_to_len - orig_seq_len
        mask[:, -pad_width:] = 0

        return x_data, y_label, mask, sample_keys

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir
        self.dataset_params = get_dataset_params(dataset_name=params.dataset_name)
        self.n_temporal_channels = self.dataset_params['n_temporal_channels']
        self.n_spatial_channels = self.dataset_params['n_spatial_channels']
        self.orig_seq_len = params.orig_seq_len
        self.tdr = params.tdr if hasattr(params, 'tdr') else 1.0

        # NOTE: This is important because it allows us to guarantee that the order is always the same,
        # invariant of advances in the original RNG state. For example, in the case we don't have this, say we
        # create our data_loader then initialize our model, the model initialization advances the RNG state before
        # we iterate through the data_loader so we will get an order A. Now suppose we don't initialize that same model,
        # then we would get an order B. Thus, we need a separate RNG that only handles the data loader. This ensures that
        # our pipeline and the Cbramod pipeline use the same train order.
        if hasattr(self.params, 'seed'):
            self.dataloader_rng = torch.Generator()
            self.dataloader_rng.manual_seed(params.seed)
        else:
            print('WARNING: Seed was not given, so train generator will not be set!!!')

        self._cached_sample_orders = {}

    def get_data_loader(self):
        train_set = CustomLMDBNumpyDataset(
            self.params.dataset_name, self.dataset_dir, mode='train', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels, 
            orig_seq_len=self.orig_seq_len, pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        val_set = CustomLMDBNumpyDataset(
            self.params.dataset_name, self.dataset_dir, mode='val', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels, 
            orig_seq_len=self.orig_seq_len, pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        test_set = CustomLMDBNumpyDataset(
            self.params.dataset_name, self.dataset_dir, mode='test', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels, 
            orig_seq_len=self.orig_seq_len, pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        train_collate_fn = partial(train_set.collate_with_mask, orig_seq_len=self.orig_seq_len) if self.params.return_mask else train_set.collate
        val_collate_fn = partial(val_set.collate_with_mask, orig_seq_len=self.orig_seq_len) if self.params.return_mask else val_set.collate
        test_collate_fn = partial(test_set.collate_with_mask, orig_seq_len=self.orig_seq_len) if self.params.return_mask else test_set.collate

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_collate_fn,
                shuffle=True,
                generator=self.dataloader_rng if hasattr(self.params, 'seed') else None
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_collate_fn,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_collate_fn,
                shuffle=False,
            ),
        }
        return data_loader