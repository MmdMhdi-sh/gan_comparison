import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.network = nn.Sequential(
            nn.Linear(
                in_features=self.latent_dim,
                out_features=256
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=256,
                out_features=512
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=1024,
                out_features=784
            ),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.network(z)
        x = x.view(-1, 1, 28, 28)
        return x
    

config = {
    "latent_dim": 100
}

generator = Generator(config=config)

z = torch.randn(16, 100)

fake_images = generator(z)

print(fake_images.shape)