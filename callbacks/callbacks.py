import torch
import os


def save_checkpoint(model, epoch, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict()
        },
        path
    )