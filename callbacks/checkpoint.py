import torch
import os


def save_checkpoint(model, epoch, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,

            "generator": model.generator.state_dict(),

            "discriminator": (
                model.discriminator.state_dict()
                if hasattr(model, "discriminator")
                else None
            ),

            "g_optimizer": model.g_optimizer.state_dict(),

            "d_optimizer": (
                model.d_optimizer.state_dict()
                if hasattr(model, "d_optimizer")
                else None
            ),

            "history": model.history if hasattr(model, "history") else None

        },
        path
    )