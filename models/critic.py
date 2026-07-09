import torch.nn as nn

from models.discriminator import Maxout

class Critic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.network = nn.Sequential(
            Maxout(784, 256, pieces=5),

            Maxout(256, 256, pieces=5),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

