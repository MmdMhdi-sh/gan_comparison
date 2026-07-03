import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
import os

# ==================================================
# Save image grids during training
# ==================================================
def save_image_grid(images, epoch, out_dir="outputs", nrows=4):
    os.makedirs(out_dir, exist_ok=True)
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    grid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), cmap="gray")
    plt.savefig(f"{out_dir}/sample_epoch_{epoch:03d}.png", bbox_inches="tight")
    plt.close