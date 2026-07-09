from algorithms.base_gan import BaseGAN

class BEGAN(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)