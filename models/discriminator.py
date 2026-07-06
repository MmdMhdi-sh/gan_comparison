import torch
import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, in_features: int, out_features: int, pieces: int = 5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features*pieces
        )

    def forward(self, x):
        x = self.linear(x)
        # (batch, out_features * pieces) -> (batch, out_features, pieces)
        x = x.view(
            x.size(0),
            self.out_features,
            self.pieces
        )

        return torch.max(x, dim=2).values

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.network(x)
    
