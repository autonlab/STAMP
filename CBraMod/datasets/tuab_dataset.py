from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.util import to_tensor
from .utils import get_dataset_params
import os
from functools import partial

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode,
            pad_to_len=0,
            reshape_data=False
    ):
        super(CustomDataset, self).__init__()

        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]

        self.pad_to_len = pad_to_len
        self.reshape_data = reshape_data
        self.mode = mode

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file)
        y_pos = file.rfind('y', 0, -3)
        label = int(file[y_pos+1:-4])

        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])

        if self.pad_to_len and x_data.shape[-1] < self.pad_to_len:
            pad_width = self.pad_to_len - x_data.shape[-1]
            x_data = np.pad(x_data, pad_width=((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant') # Pad to (batch_size, n_spatial_channels, n_temporal_channels, pad_to_len)

        if self.reshape_data:
            x_data = x_data.reshape(-1, x_data.shape[-1]) # Reshape to (batch_size * n_spatial_channels * n_temporal_channels, seq_len)
            x_data = np.expand_dims(x_data, axis=1) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, 1, seq_len)

        return to_tensor(x_data), to_tensor(y_label).long()

    def collate_with_mask(dataset, batch, orig_seq_len):
        x_data, y_label = dataset.collate(batch)

        # Create a mask for the sequence length
        mask = torch.ones(x_data.shape[0], x_data.shape[-1], dtype=torch.bool)
        # Zero out the padding part of the mask
        pad_width = dataset.pad_to_len - orig_seq_len
        mask[:, -pad_width:] = 0

        return x_data, y_label, mask

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir
        self.dataset_params = get_dataset_params(dataset_name=params.dataset_name)
        self.n_temporal_channels = self.dataset_params['n_temporal_channels']
        self.n_spatial_channels = self.dataset_params['n_spatial_channels']
        self.orig_seq_len = params.orig_seq_len

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
        train_set = CustomDataset(self.dataset_dir, mode='train', pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data)
        val_set = CustomDataset(self.dataset_dir, mode='val', pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data)
        test_set = CustomDataset(self.dataset_dir, mode='test', pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))

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