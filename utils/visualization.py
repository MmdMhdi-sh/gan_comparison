import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
import os

# ==================================================
# Save image grids during training
# ==================================================
def save_image_grid(images, epoch, out_dir="outputs", nrows=4):
    os.makedirs(out_dir, exist_ok=True)
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, value_range=(0,1))
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    grid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), cmap="gray")
    plt.savefig(f"{out_dir}/sample_epoch_{epoch:03d}.png", bbox_inches="tight")
    plt.close()


# ==================================================
# Plot all recorded metrics
# ==================================================
def plot_history(history, save_dir, groups=None):
    # Get dictionary if a History object is passed
    if hasattr(history, "get"):
        history = history.get()

    # Make a directory for saving figures
    os.makedirs(save_dir, exist_ok=True)

    # Default: plot everything together
    if groups is None:
        groups = [list(history.keys())]

    for group in groups:

        plt.figure(figsize=(8, 5))

        for metric in group:
            if metric in history:
                plt.plot(history[metric], label=metric)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        title = "_".join(group)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()