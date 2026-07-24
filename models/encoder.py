import torch.nn as nn

from models.decoder import DecoderConv

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ELU(inplace=True),

            nn.Linear(256, 128),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

class GeneratorConv(nn.Module):
    def __init__(self, config, n=64):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.decoder = DecoderConv(latent_dim=self.latent_dim, n=n)

    def forward(self, z):
        return self.decoder(z)

