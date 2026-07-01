from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import(
    load_dataloaders,
    sample_noise
)

if __name__ == "__main__":
    train_data, test_data, train_dataloader, test_dataloader = load_dataloaders(root="./data")
