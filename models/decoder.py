import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(inplace=True),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
    

class DecoderConv(nn.Module):
    def __init__(self, embedding_dim=64, n=64):
        super().__init__()
        self.n = n
        self.fc = nn.Linear(embedding_dim, n * 2 * 7 * 7)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 7 -> 14
            nn.Conv2d(n * 2, n, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 14 -> 28
            nn.Conv2d(n, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h):
        x = self.fc(h)
        x = x.view(x.size(0), self.n * 2, 7, 7)
        return self.net(x)