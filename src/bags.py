"""Variable-length MIL bags and mini-batch collation."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BagDataset(Dataset):
    def __init__(self, features, offsets, labels, users, excluded=None):
        self.features = features
        self.offsets = offsets
        self.labels = labels
        self.users = list(users)
        self.excluded = excluded or {}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        user = self.users[item]
        start, end = self.offsets[user]
        bag = self.features[start:end]
        removed = self.excluded.get(user)
        if removed is not None and len(removed):
            keep = np.ones(end - start, dtype=bool)
            keep[removed] = False
            bag = bag[keep]
        return bag.toarray().astype(np.float32, copy=False), int(self.labels[user])


def collate_bags(batch):
    arrays, labels = zip(*batch)
    maximum = max(map(len, arrays))
    padded = np.zeros((len(arrays), maximum, arrays[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(arrays), maximum), dtype=bool)
    for index, array in enumerate(arrays):
        padded[index, : len(array)] = array
        mask[index, : len(array)] = True
    return torch.from_numpy(padded), torch.from_numpy(mask), torch.tensor(labels)


def make_loader(
    features, offsets, labels, users, batch_size, shuffle=False, excluded=None
):
    return DataLoader(
        BagDataset(features, offsets, labels, users, excluded),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_bags,
    )
