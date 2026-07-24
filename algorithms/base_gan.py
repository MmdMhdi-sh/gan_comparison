from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.generator import Generator, GeneratorConv
from utils.optimizer import build_optimizer

class BaseGAN(nn.Module, ABC):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device
        self.latent_dim = config["latent_dim"]

        self.generator = GeneratorConv(config).to(device)
    
    def sample_noise(self, batch_size):
        # (batch_size, latent_dim) samples
        return torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device
        )
    
    # Generate fake images
    def generate(self, z=None, n_samples: int = 16):
        self.generator.eval()

        with torch.no_grad():
            if z is None:
                z = self.sample_noise(n_samples)
            
            images = self.generator(z)

        return images.cpu()
    
    @staticmethod
    def set_requires_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def build_optimizer(parameters, config):
        return build_optimizer(parameters, config)
    
    @property
    def plot_groups(self):
        raise NotImplementedError
    
    @property
    def checkpoint(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_checkpoint(self, checkpoint):
        raise NotImplementedError