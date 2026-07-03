import torch
import torch.nn as nn
import torch.optim as optim

from ..models.discriminator import Discriminator
from ..models.generator import Generator
from ..models.weights import initialize_weights

class NaiveGAN(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.latent_dim = config["latent_dim"]

        self.generator = Generator(config).to(device)
        self.discriminator = Discriminator().to(device)

        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

        self.g_optimizer = optim.SGD(
            self.generator.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"]
        )
        self.d_optimizer = optim.SGD(
            self.discriminator.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"]
        )

        self.criterion = nn.BCELoss()

        self.device = device
    
    def sample_noise(self, batch_size):
        # (batch_size, latent_dim) samples
        return torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device
        )
    
    def update_discriminator(self, real_images):
        self.discriminator.train()
        self.generator.train()

        real_images = real_images.to(self.device)

        z = self.sample_noise(real_images.size(0))
        fake_images = self.generator(z).detach()

        real_preds = self.discriminator(real_images)
        real_labels = torch.ones_like(real_preds)

        fake_preds = self.discriminator(fake_images)
        target_labels = torch.zeros_like(fake_preds)

        real_loss = self.criterion(real_preds, real_labels)
        fake_loss = self.criterion(fake_preds, target_labels)

        d_loss = real_loss + fake_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()

        self.d_optimizer.step()
        

        return d_loss.item()

    def update_generator(self, batch_size):
        self.generator.train()
        self.discriminator.train()

        for p in self.discriminator.parameters():
            p.requires_grad = False

        z = self.sample_noise(batch_size)
        fake_images = self.generator(z)

        discriminator_preds = self.discriminator(fake_images)
        fake_labels = torch.ones_like(discriminator_preds)

        g_loss = self.criterion(discriminator_preds, fake_labels)

        self.g_optimizer.zero_grad()

        g_loss.backward()

        self.g_optimizer.step()

        return g_loss.item()

    def train_step(self, real_images):
        d_loss = self.update_discriminator(real_images)

        g_loss = self.update_generator(real_images.size(0))

        return d_loss, g_loss

    # Generate fake images
    def generate(self, n_samples):
        self.generator.eval()

        with torch.no_grad():
            z = self.sample_noise(n_samples)
            
            images = self.generator(z)

        return images
