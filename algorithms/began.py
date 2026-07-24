import torch

from algorithms.base_gan import BaseGAN

from models.autoencoder import AutoEncoder, AutoEncoderConv
from models.weights import initialize_weights

class BEGAN(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.discriminator = AutoEncoderConv().to(device)

        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

        self.g_optimizer = self.build_optimizer(
            self.generator.parameters(),
            config
        )

        self.d_optimizer = self.build_optimizer(
            self.discriminator.parameters(),
            config
        )

        self.gamma = config["gamma"]
        self.lambda_k = config["lambda_k"]

        self.kt = 0.0

        self.lr_decay_patience = config.get("lr_decay_patience", 5)
        self.lr_decay_factor = config.get("lr_decay_factor", 0.5)
        self.lr_decay_threshold = config.get("lr_decay_threshold", 1e-4)

        self.best_M_global = float("inf")
        self.epochs_since_improvement = 0

    def end_epoch(self, avg_M_global):
        if avg_M_global < self.best_M_global - self.lr_decay_threshold:
            self.best_M_global = avg_M_global
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.lr_decay_patience:
            self._decay_lr()
            self.epochs_since_improvement = 0

    def _decay_lr(self):
        for optimizer in (self.g_optimizer, self.d_optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.lr_decay_factor
        print(f"[BEGAN] M_global plateaued — LR decayed by {self.lr_decay_factor}")

    @staticmethod
    def reconstruction_loss(x, x_hat):
        return torch.mean(torch.abs(x - x_hat))
    
    def update_discriminator(self, real_images):
        self.generator.train()
        self.discriminator.train()

        real_images = real_images.to(self.device)

        z = self.sample_noise(real_images.size(0))

        fake_images = self.generator(z).detach()

        real_recon = self.discriminator(real_images)
        fake_recon = self.discriminator(fake_images)

        real_loss = self.reconstruction_loss(real_images, real_recon)
        fake_loss = self.reconstruction_loss(fake_images, fake_recon)

        discriminator_loss = real_loss - self.kt * fake_loss
        
        self.d_optimizer.zero_grad()
        discriminator_loss.backward()
        self.d_optimizer.step()

        return {
            "discriminator_loss": discriminator_loss.item(),
            "real_loss": real_loss.item(),
            "fake_loss": fake_loss.item()
        }
    
    def update_generator(self, batch_size):
        self.generator.train()
        self.discriminator.train()

        z = self.sample_noise(batch_size)

        self.set_requires_grad(self.discriminator, False)

        fake_images = self.generator(z)
        fake_recon = self.discriminator(fake_images)

        generator_loss = self.reconstruction_loss(fake_images, fake_recon)

        self.g_optimizer.zero_grad()
        generator_loss.backward()
        self.g_optimizer.step()

        self.set_requires_grad(self.discriminator, True)
        
        with torch.no_grad():
            diversity = self.batch_diversity(fake_images).item()

        return generator_loss.item(), diversity
    
    def update_kt(self, real_loss, fake_loss):
        self.kt = self.kt + self.lambda_k * (self.gamma * real_loss - fake_loss)
        self.kt = min(max(self.kt, 0.0), 1.0)

    def train_step(self, real_images):
        d_logs = self.update_discriminator(real_images)

        generator_loss, diversity  = self.update_generator(real_images.size(0))

        self.update_kt(d_logs["real_loss"], d_logs["fake_loss"])

        M_global = d_logs["real_loss"] + abs(
            self.gamma * d_logs["real_loss"] - d_logs["fake_loss"]
        )

        return {
                "reconstruction_loss": d_logs["real_loss"],
                "generator_loss": generator_loss,
                "M_global": M_global,
                "kt": self.kt,
                "diversity": diversity
            }

    @property
    def plot_groups(self):
        return [
            ["reconstruction_loss", "generator_loss"],
            ["kt"],
            ["M_global"],
            ["diversity"]
        ]
    
    @property
    def checkpoint(self):
        return {
            "config": self.config,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "kt": self.kt,
        }
    
    def load_checkpoint(self, checkpoint):
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        self.kt = checkpoint.get("kt", 0.0)

    @staticmethod
    def batch_diversity(images):
        # Mean pairwise L1 distance across a batch, flattened per-image.
        # Higher = more diverse; near-zero = collapse.
        flat = images.view(images.size(0), -1)
        diffs = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0), p=1).squeeze(0)
        n = images.size(0)
        # Exclude the zero diagonal (each image vs itself)
        return diffs.sum() / (n * (n - 1))