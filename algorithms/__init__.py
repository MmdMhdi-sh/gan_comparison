from .began import BEGAN
from .naive_gan import NaiveGAN
from .wgan_gp import WGANGP

def build_model(config, device):
    algorithm = config["algorithm"].lower()

    if algorithm == "naive_gan":
        return NaiveGAN(config, device)
    
    elif algorithm == "wgan_gp":
        return WGANGP(config, device)
    
    elif algorithm == "began":
        return BEGAN(config, device)
    
    raise ValueError(f"Unknown algorithm: {algorithm}")