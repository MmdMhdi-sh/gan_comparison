import torch.nn as nn

from models.decoder import DecoderConv

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 1200),
            nn.ELU(inplace=True),
            nn.Linear(1200, 784),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # (batch_size, latent_dim) -> (batch_size, 1, 28, 28)
        x = self.network(z)
        x = x.view(-1, 1, 28, 28)
        return x
    

class GeneratorConv(nn.Module):
    def __init__(self, config, n=32):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.n = n

        self.fc = nn.Linear(self.latent_dim, n * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 7 -> 14
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.2),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 14 -> 28
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(n, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.n, 7, 7)
        return self.net(x)