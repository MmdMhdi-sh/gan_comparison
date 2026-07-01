import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

def load_dataloaders(root: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        datset=train_data,
        batch_size=16,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=16,
        shuffle=True
    )

    return train_data, test_data, train_dataloader, test_dataloader


# ==================================================
# Sample from normal distrubution with dimension of (batch_size, latent_dim)
# ==================================================
def sample_noise(batch_size: int, latent_dim: int):
    return torch.randn(batch_size, latent_dim)