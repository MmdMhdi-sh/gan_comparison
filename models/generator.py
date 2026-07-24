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
    def __init__(self, config, n=64):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.decoder = DecoderConv(latent_dim=self.latent_dim, n=n)

    def forward(self, z):
        return self.decoder(z)