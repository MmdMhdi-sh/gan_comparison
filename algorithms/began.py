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