import torch
from docopt import docopt
from ruamel.yaml import YAML

from trainer import PPOTrainer


def _load_config(path: str) -> dict:
    """Load the YAML config file and return its contents as a dict."""
    yaml = YAML()
    with open(path, "r", encoding="utf-8") as stream:
        config = {}
        for data in yaml.load_all(stream):
            if data:
                config = dict(data)
    if not config:
        raise ValueError(f"Config file '{path}' did not contain any data.")
    return config


def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/cartpole.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = _load_config(options["--config"])

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()


if __name__ == "__main__":
    main()
