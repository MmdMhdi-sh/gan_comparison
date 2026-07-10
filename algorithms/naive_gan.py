import torch
import torch.nn as nn
import torch.optim as optim

from utils.optimizer import build_optimizer

from algorithms.base_gan import BaseGAN
from models.discriminator import Discriminator
from models.generator import Generator
from models.weights import initialize_weights


class NaiveGAN(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.discriminator = Discriminator().to(device)

        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

        self.d_steps = config.get("d_steps", 1)
        self.g_steps = config.get("g_steps", 1)

        self.g_optimizer = build_optimizer(
            self.generator.parameters(),
            config
        )

        self.d_optimizer = self.build_optimizer(
            self.discriminator.parameters(),
            config
        )

        self.criterion = nn.BCEWithLogitsLoss()


    def update_discriminator(self, real_images):
        self.discriminator.train()
        self.generator.train()

        self.set_requires_grad(self.discriminator, True)

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

        self.set_requires_grad(self.discriminator, False)

        z = self.sample_noise(batch_size)
        fake_images = self.generator(z)

        discriminator_preds = self.discriminator(fake_images)
        fake_labels = torch.ones_like(discriminator_preds)

        g_loss = self.criterion(discriminator_preds, fake_labels)

        self.g_optimizer.zero_grad()

        g_loss.backward()

        self.g_optimizer.step()

        self.set_requires_grad(self.discriminator, True)

        return g_loss.item()

    def train_step(self, real_images):
        d_losses = []
        for _ in range(self.d_steps):
            d_losses.append(self.update_discriminator(real_images))

        g_losses = []
        for _ in range(self.g_steps):
            g_losses.append(self.update_generator(real_images.size(0)))

        return {
            "d_loss": sum(d_losses) / len(d_losses),
            "g_loss": sum(g_losses) / len(g_losses),
        }
    
    @property
    def plot_groups(self):
        return [
            ["d_loss", "g_loss"]
        ]
    
    @property
    def checkpoint(self):
        return {
            "config": self.config,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint):

        self.generator.load_state_dict(
            checkpoint["generator"]
        )

        self.discriminator.load_state_dict(
            checkpoint["discriminator"]
        )

        self.g_optimizer.load_state_dict(
            checkpoint["g_optimizer"]
        )

        self.d_optimizer.load_state_dict(
            checkpoint["d_optimizer"]
        )
# class NaiveGAN(nn.Module):
#     def __init__(self, config, device):
#         super().__init__()
#         self.latent_dim = config["latent_dim"]

#         self.generator = Generator(config).to(device)
#         self.discriminator = Discriminator().to(device)

#         self.d_steps = config.get("d_steps", 1)
#         self.g_steps = config.get("g_steps", 1)

#         initialize_weights(self.generator)
#         initialize_weights(self.discriminator)

#         self.g_optimizer = build_optimizer(
#             self.generator.parameters(),
#             config
#         )
        
#         self.d_optimizer = build_optimizer(
#             self.discriminator.parameters(),
#             config
#         )

#         self.criterion = nn.BCEWithLogitsLoss()

#         self.device = device
    
#     def sample_noise(self, batch_size):
#         # (batch_size, latent_dim) samples
#         return torch.randn(
#             batch_size,
#             self.latent_dim,
#             device=self.device
#         )
    
#     @staticmethod
#     def set_requires_grad(model, requires_grad):
#         for p in model.parameters():
#             p.requires_grad = requires_grad
    
#     def update_discriminator(self, real_images):
#         self.discriminator.train()
#         self.generator.train()

#         self.set_requires_grad(self.discriminator, True)

#         real_images = real_images.to(self.device)

#         z = self.sample_noise(real_images.size(0))
#         fake_images = self.generator(z).detach()


#         real_preds = self.discriminator(real_images)
#         real_labels = torch.ones_like(real_preds) 

#         fake_preds = self.discriminator(fake_images)
#         target_labels = torch.zeros_like(fake_preds)

#         real_loss = self.criterion(real_preds, real_labels)
#         fake_loss = self.criterion(fake_preds, target_labels)

#         d_loss = real_loss + fake_loss

#         self.d_optimizer.zero_grad()
#         d_loss.backward()

#         self.d_optimizer.step()
        
#         print(
#             f"D(real): {torch.sigmoid(real_preds).mean().item():.4f} "
#             f"D(fake): {torch.sigmoid(fake_preds).mean().item():.4f}"
#         )

#         return d_loss.item()

#     def update_generator(self, batch_size):
#         self.generator.train()
#         self.discriminator.train()

#         self.set_requires_grad(self.discriminator, False)

#         z = self.sample_noise(batch_size)
#         fake_images = self.generator(z)

#         discriminator_preds = self.discriminator(fake_images)
#         fake_labels = torch.ones_like(discriminator_preds)

#         g_loss = self.criterion(discriminator_preds, fake_labels)

#         self.g_optimizer.zero_grad()

#         g_loss.backward()

#         self.g_optimizer.step()

#         self.set_requires_grad(self.discriminator, True)

#         return g_loss.item()

#     def train_step(self, real_images):
#         d_losses = []
#         for _ in range(self.d_steps):
#             d_losses.append(self.update_discriminator(real_images))

#         g_losses = []
#         for _ in range(self.g_steps):
#             g_losses.append(self.update_generator(real_images.size(0)))

#         return {
#             "d_loss": sum(d_losses) / len(d_losses),
#             "g_loss": sum(g_losses) / len(g_losses),
#         }

#     # Generate fake images
#     def generate(self, z=None, n_samples: int = 16):
#         self.generator.eval()

#         with torch.no_grad():
#             if z is None:
#                 z = self.sample_noise(n_samples)
            
#             images = self.generator(z)

#         return images.cpu()
