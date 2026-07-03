# =============================================
# Imports
# =============================================
import time 

import torch

from tqdm import tqdm

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
model = NaiveGAN(
    config=config, 
    device=device
)

# =============================================
# Training
# =============================================
def main():
    start_time = time.time()
    print("=" * 50)
    print(f"Training Starts ...")
    num_epochs = config["epochs"]
    for epoch in tqdm(
        range(num_epochs),
        desc="Training"
    ):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for real_images, _ in train_loader:
            losses = model.train_step(real_images)

            epoch_d_loss += losses["d_loss"]
            epoch_g_loss += losses["g_loss"]

        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Discriminator Loss: {epoch_d_loss:.4f} "
            f"Generator Loss: {epoch_g_loss:.4f} "
            f"Duration: {time.time()-start_time:.2f}"
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = model.generate(n_samples=16)
            save_image_grid(samples, epoch + 1, out_dir="outputs", nrows=4)

if __name__ == "__main__":
    main()