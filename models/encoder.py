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
    def __init__(self, embedding_dim=64, n=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n, kernel_size=3, stride=2, padding=1),       # 28 -> 14
            nn.ELU(inplace=True),

            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1),   # 14 -> 7
            nn.ELU(inplace=True),
        )
        self.fc = nn.Linear(n * 2 * 7 * 7, embedding_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

