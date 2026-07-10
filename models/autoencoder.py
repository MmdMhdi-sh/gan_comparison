import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(
            x.size(0),
            -1
        )

        h = self.encoder(x)
        x_hat = self.decoder(h)

        return x_hat.view(-1, 1, 28, 28)