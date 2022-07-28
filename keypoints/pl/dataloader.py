from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import MatDataset


def train_collate_fn(batch):
    data_list, label_list = [], []

    for data, label in batch:
        data_list.append(data)
        label_list.append(label)

    return data_list, label_list


def test_collate_fn(batch):
    data_list = []

    for data in batch:
        data_list.append(data)

    return data_list


class MatDataloader(pl.LightningDataModule):
    def __init__(self, folder: str, batch_size: int = 4):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f"{str(folder.resolve())} is not dir!"
        self.batch_size = batch_size
        # self.test_dataset = MatDataset(root_folder=folder)

    def train_dataloader(self):
        return DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=train_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=train_collate_fn
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=test_collate_fn
    #     )
