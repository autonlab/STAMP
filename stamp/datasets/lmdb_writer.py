import os
import lmdb
import numpy as np
import json

class LMDBWriter:
    def __init__(self, lmdb_path, map_size):
        """Initialize LMDB writer

        Args:
            lmdb_path: Path to LMDB database
            map_size: Maximum size of database in bytes
        """
        self.lmdb_path = lmdb_path
        self.map_size = map_size
        self.env = None
        self.txn = None
        self.count = 0
        self.keys = []

    def __enter__(self):
        # Create directory if it doesn't exist
        os.makedirs(self.lmdb_path, exist_ok=True)

        # Open LMDB environment
        self.env = lmdb.open(
            self.lmdb_path,
            map_size=self.map_size,
        )
        self.txn = self.env.begin(write=True)
        return self

    def write_sample(self, sample, label, filename, dtype):
        """Write a single sample to LMDB

        Args:
            signal: numpy array of EEG signal data
            metadata: dict containing label, offending_channel, filename, etc.
        """

        sample_bytes = sample.astype(dtype).tobytes()
        key = f'{filename}_{self.count:08d}_y{label}'.encode()
        self.txn.put(key, sample_bytes)
        self.keys.append(key)

        self.count += 1

        # Commit every 1000 samples to avoid memory issues
        if self.count % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def get_count(self):
        return self.count

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.txn:
                if exc_type is None:  # Only write metadata if no exception occurred
                    # Store metadata before final commit
                    self.txn.put(b'__count__', str(self.count).encode())

                    # Store keys list as JSON for fast loading
                    keys_str = [k.decode() for k in self.keys]
                    self.txn.put(b'__keys__', json.dumps(keys_str).encode())

                    self.txn.commit()
                else:
                    # If there was an exception, abort the transaction
                    print(f'there was an exception, abort the transaction {exc_type}')
                    self.txn.abort()
        except Exception as e:
            print(f"Error in __exit__: {e}")
            if self.txn:
                try:
                    self.txn.abort()
                except:
                    pass  # Transaction might already be invalid
        finally:
            if self.env:
                self.env.close()