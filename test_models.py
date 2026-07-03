import torch

from models.generator import Generator
from models.discriminator import Discriminator

config = {
    "latent_dim": 100
}
generator = Generator(config)
discriminator = Discriminator()

z = torch.randn(16, config["latent_dim"])

fake_images = generator(z)
output = discriminator(fake_images)

print(fake_images.shape)
print(output.shape)

print(output.min().item())
print(output.max().item())