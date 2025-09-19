import torch
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
    run_id = "run"
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = _load_config("./minigrid.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()


if __name__ == "__main__":
    main()
