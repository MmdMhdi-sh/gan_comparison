import torch.nn as nn

from models.encoder import Encoder, EncoderConv
from models.decoder import Decoder, DecoderConv

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
    

class AutoEncoderConv(nn.Module):
    def __init__(self, latent_dim=128, n=32):
        super().__init__()

        self.encoder = EncoderConv(latent_dim, n)
        self.decoder = DecoderConv(latent_dim, n)

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat