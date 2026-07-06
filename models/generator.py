import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 784),
            nn.Sigmoid()
        )
        # self.network = nn.Sequential(
        #     nn.Linear(self.latent_dim, 256),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(256, 512),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(1024, 784),
        #     nn.Sigmoid()
        # )
    
    def forward(self, z):
        # (batch_size, latent_dim) -> (batch_size, 1, 28, 28)
        x = self.network(z)
        x = x.view(-1, 1, 28, 28)
        return x
    
