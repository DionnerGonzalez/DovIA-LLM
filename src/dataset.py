"""
DovIA v2 - Dataset con ventana deslizante.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union
from src.tokenizer import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BPETokenizer,
                 context_length: int = 512, stride: int = 256):
        self.context_length = context_length
        self.samples = []
        print(f"[DovIA Dataset] Tokenizando {len(texts)} textos...")
        all_ids = []
        for t in texts:
            all_ids.extend(tokenizer.encode(t, add_special_tokens=True))
        for i in range(0, len(all_ids) - context_length, stride):
            chunk = all_ids[i: i + context_length + 1]
            if len(chunk) == context_length + 1:
                self.samples.append(chunk)
        print(f"[DovIA Dataset] {len(self.samples)} muestras generadas.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        t = torch.tensor(self.samples[idx], dtype=torch.long)
        return t[:-1], t[1:]


def get_dataloader(dataset, batch_size=16, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())
