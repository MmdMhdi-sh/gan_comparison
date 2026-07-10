# =============================================
# Imports
# =============================================
import argparse
import time 

import torch

from tqdm import tqdm

from algorithms import build_model
from data.datamodule import DataModule
from callbacks.checkpoint import save_checkpoint
from utils.configs import load_config
from utils.history import History
from utils.visualization import save_image_grid, plot_history

# =============================================
# Device & Config
# =============================================
torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file."
    )

    return parser.parse_args()

args = parse_args()

config = load_config(args.config)

print(f"Using {device}\n"
      f"Config loaded from {args.config}"
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
    history = History()
    
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
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        history.update(epoch_losses)

        message = f"Epoch [{epoch+1}/{num_epochs}] "

        for key, value in epoch_losses.items():
            message += f"{key}: {value:.4f} "

        message += f"Duration: {time.time()-start_time:.2f}s"

        print(message)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = model.generate(z=fixed_noise, n_samples=16)
            save_image_grid(samples, epoch + 1, out_dir="outputs", nrows=4)
            checkpoint_path = f"outputs/{config['algorithm']}/checkpoints"
            save_checkpoint(
                model,
                epoch,
                checkpoint_path
            )

    print("="*50)
    print("Plotting Figures ...")
    save_plot_path = f'outputs/{config["algorithm"]}/plots'
    plot_history(
        history, 
        save_plot_path,
        groups=model.plot_groups
    )
    print(f"Figures Saved at {save_plot_path}")

if __name__ == "__main__":
    main()