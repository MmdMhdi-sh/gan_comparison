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

class EncoderConv(nn.Module):
    def __init__(self, latent_dim=128, n=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(n, 2 * n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.AvgPool2d(2),  # 28 -> 14

            nn.Conv2d(2 * n, 2 * n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(2 * n, 3 * n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.AvgPool2d(2),  # 14 -> 7

            nn.Conv2d(3 * n, 3 * n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(3 * n, 3 * n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
        )
        self.fc = nn.Linear(3 * n * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

