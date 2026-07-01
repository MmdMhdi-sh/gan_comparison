from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataModule:
    def __init__(self, config):
        self.config = config

        self.train_ds = None
        self.test_ds = None

    def setup(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_ds = datasets.MNIST(
            root=self.config["data_dir"],
            train=True,
            download=True,
            transform=transform
        )

        self.test = datasets.MNIST(
            root=self.config["data_dir"],
            train=False,
            download=True,
            transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.config["batch_size"],
            shuffle=False
        )
