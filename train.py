# =============================================
# Imports
# =============================================
import time 

import torch

from tqdm import tqdm

from algorithms import build_model
from algorithms.naive_gan import NaiveGAN
from data.datamodule import DataModule
from utils.configs import load_config
from utils.visualization import save_image_grid

# =============================================
# Device & Config
# =============================================
torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)


path = "configs/naive_gan.yaml"
config = load_config(path)
print(f"Using {device}\n"
      f"Config loaded from {path}"
)
# =============================================
# Data loaders
# =============================================
data_module = DataModule(config)

data_module.setup()
train_loader = data_module.train_dataloader()

# =============================================
# Model
# =============================================
model = build_model(
    config=config, 
    device=device
)

# =============================================
# Training
# =============================================
def main():
    history = {}
    
    print("=" * 50)
    print(f"Training Starts ...")
    num_epochs = config["epochs"]
    fixed_noise = model.sample_noise(16)
    for epoch in tqdm(
        range(num_epochs),
        desc="Training"
    ):
        start_time = time.time()
        epoch_losses = {}
        for real_images, _ in train_loader:
            losses = model.train_step(real_images)

            for key, value in losses.items():
                history.setdefault(key, []).append(value)

        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            history.setdefault(key, []).append(epoch_losses[key])

        message = f"Epoch [{epoch+1}/{num_epochs}] "

        for key, value in epoch_losses.items():
            message += f"{key}: {value:.4f} "

        message += f"Duration: {time.time()-start_time:.2f}s"

        print(message)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = model.generate(z=fixed_noise, n_samples=16)
            save_image_grid(samples, epoch + 1, out_dir="outputs", nrows=4)
    

    z1 = torch.randn(1,100,device=device)
    z2 = torch.randn(1,100,device=device)

    x1 = model.generator(z1)
    x2 = model.generator(z2)

    print("="*50)
    print(f"x1-x2: {torch.mean(torch.abs(x1-x2))}")

if __name__ == "__main__":
    main()