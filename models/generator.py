import torch.nn as nn

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
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config["latent_dim"]

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.net(x)
        return x