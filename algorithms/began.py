from algorithms.base_gan import BaseGAN

class BEGAN(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def plot_groups(self):
        return [
            ["reconstruction_loss", "generator_loss"],
            ["kt"],
            ["M_global"]
        ]
    
    @property
    def checkpoint(self):
        return {
            "config": self.config,
            "generator": self.generator.state_dict(),
            "autoencoder": self.autoencoder.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "ae_optimizer": self.ae_optimizer.state_dict(),
            "kt": self.kt,
        }