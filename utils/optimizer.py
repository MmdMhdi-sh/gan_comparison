import torch.optim as optim

def build_optimizer(parameters, optimizer_config):

    name = optimizer_config["name"].lower()

    if name == "adam":
        return optim.Adam(
            parameters,
            lr=optimizer_config["lr"],
            betas=tuple(optimizer_config["betas"])
        )

    elif name == "sgd":
        return optim.SGD(
            parameters,
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"]
        )

    elif name == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=optimizer_config["lr"]
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")