from algorithms.base_gan import BaseGAN

class WGANGP(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def plot_groups(self):
        return [
            ["critic_loss", "generator_loss"],
            ["gradient_penalty"]
        ]