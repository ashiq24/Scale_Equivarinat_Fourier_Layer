import torchaudio
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class YESNODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        yesno_dataset = torchaudio.datasets.YESNO(self.data_dir, download=True)
        dataset_size = len(yesno_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        self.yesno_train, self.yesno_val = random_split(
            yesno_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.yesno_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.yesno_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.yesno_val, batch_size=self.batch_size, num_workers=4)
