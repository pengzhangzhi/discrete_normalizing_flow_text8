import os
import pickle
import zipfile

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        seq = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        seq_onehot = F.one_hot(seq, self.vocab_size).float()
        return (seq_onehot,)


def get_dataloaders(
    root: str = "./data/text8",
    batch_size: int = 512,
    seq_len: int = 256,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = Text8Dataset(root, "train", seq_len=seq_len)
    val_dataset = Text8Dataset(root, "valid", seq_len=seq_len)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
