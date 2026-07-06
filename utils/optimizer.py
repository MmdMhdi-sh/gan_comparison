import torch.optim as optim

def build_optimizer(parameters, config):

    name = config["optimizer"]["name"].lower()

    if name == "adam":
        return optim.Adam(
            parameters,
            lr=config["optimizer"]["lr"],
            betas=tuple(config["optimizer"]["betas"])
        )

    elif name == "sgd":
        return optim.SGD(
            parameters,
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"]
        )

    elif name == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=config["optimizer"]["lr"]
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")