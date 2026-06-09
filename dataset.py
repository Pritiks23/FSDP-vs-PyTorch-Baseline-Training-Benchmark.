import torch
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, size=2000, seq_len=64):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, 50257, (self.seq_len,))
        return {"input_ids": x, "labels": x.clone()}
