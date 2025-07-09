# +
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义 mRNA 序列的字符映射
BASE_MAPPING = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


class MRNADataset(Dataset):
    def __init__(self, sequences, half_lives):
        self.sequences = sequences
        self.half_lives = half_lives

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        half_life = self.half_lives[idx]

        # 将 mRNA 序列转换为数字编码
        encoded_sequence = [BASE_MAPPING[base] for base in sequence]
        encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.long)
        half_life = torch.tensor(half_life, dtype=torch.float32)

        return encoded_sequence, half_life


def get_dataloader(sequences, half_lives, batch_size=32, shuffle=True):
    dataset = MRNADataset(sequences, half_lives)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
