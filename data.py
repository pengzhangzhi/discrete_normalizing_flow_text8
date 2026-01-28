import os
import pickle
import zipfile

import numpy as np
import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


def prepare_text8(root: str) -> None:
    """Download and prepare the text8 dataset."""
    os.makedirs(root, exist_ok=True)
    zip_fname = os.path.join(root, "text8.zip")
    
    if not os.path.exists(os.path.join(root, "train.bin")):
        # Download
        if not os.path.exists(zip_fname):
            data_url = "http://mattmahoney.net/dc/text8.zip"
            print("Downloading text8...")
            with open(zip_fname, "wb") as f:
                f.write(requests.get(data_url).content)
            print("Done!")
        
        # Extract
        with zipfile.ZipFile(zip_fname) as f:
            f.extractall(root)
        
        # Read and process
        with open(os.path.join(root, "text8"), "r") as f:
            data = f.read()
        
        # Get unique characters
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print(f"Unique characters: {''.join(chars)}")
        print(f"Vocab size: {vocab_size}")
        
        # Create mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        def encode(s):
            return [stoi[c] for c in s]
        
        # Split data
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) : int(n * 0.95)]
        test_data = data[int(n * 0.95) :]
        
        train_ids = np.array(encode(train_data), dtype=np.uint16)
        val_ids = np.array(encode(val_data), dtype=np.uint16)
        test_ids = np.array(encode(test_data), dtype=np.uint16)
        
        print(f"Train: {len(train_ids):,} tokens")
        print(f"Val: {len(val_ids):,} tokens")
        print(f"Test: {len(test_ids):,} tokens")
        
        # Save binary files
        train_ids.tofile(os.path.join(root, "train.bin"))
        val_ids.tofile(os.path.join(root, "valid.bin"))
        test_ids.tofile(os.path.join(root, "test.bin"))
        
        # Save metadata
        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
        with open(os.path.join(root, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        
        # Cleanup
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        
        print(f"Text8 prepared in {root}")


class Text8Dataset(Dataset):
    """Text8 character-level dataset returning one-hot encoded sequences."""
    
    def __init__(self, root: str, split: str, vocab_size: int = 27, seq_len: int = 256):
        self.root = root
        self.split = split
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Map split names
        split_map = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
        fname = os.path.join(root, f"{split_map[split]}.bin")
        
        if not os.path.exists(fname):
            prepare_text8(root)
        
        self.data = np.memmap(fname, dtype=np.uint16, mode="r")
    
    def __len__(self) -> int:
        return self.data.size - self.seq_len
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        seq_onehot = F.one_hot(seq, self.vocab_size).float()
        return seq_onehot, seq


class Text8DataModule(pl.LightningDataModule):
    """LightningDataModule with StatefulDataLoader for precise checkpoint resumption."""
    
    def __init__(self, root: str = "./data/text8", batch_size: int = 512, seq_len: int = 256, num_workers: int = 4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self._train_loader = None
        self._val_loader = None
    
    def setup(self, stage: str = None):
        self.train_dataset = Text8Dataset(self.root, "train", seq_len=self.seq_len)
        self.val_dataset = Text8Dataset(self.root, "valid", seq_len=self.seq_len)
    
    def train_dataloader(self):
        self._train_loader = StatefulDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self._train_loader
    
    def val_dataloader(self):
        self._val_loader = StatefulDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return self._val_loader
    
    def state_dict(self):
        """Save dataloader state for checkpoint. Called by Lightning automatically."""
        state = {}
        if self._train_loader is not None:
            state["train_dataloader"] = self._train_loader.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Restore dataloader state from checkpoint. Called by Lightning automatically."""
        self._pending_state = state_dict.get("train_dataloader")
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Restore state after first batch (dataloader now exists)."""
        if hasattr(self, "_pending_state") and self._pending_state is not None:
            if self._train_loader is not None:
                self._train_loader.load_state_dict(self._pending_state)
            self._pending_state = None
        return batch
