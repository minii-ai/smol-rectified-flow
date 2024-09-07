import argparse
import json
from torchvision.transforms import v2

import sys

sys.path.append("./")

from rectified_flow import RectifiedFlow, RectifiedFlowTrainer
from rectified_flow.utils import get_device
from torchvision.datasets import MNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )

    return parser.parse_args()


def load_json_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_dataset():
    dataset = MNIST(
        root="./data",
        download=True,
        train=True,
        transform=v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize((32, 32)),
            ]
        ),
    )
    return dataset


def main(args):
    config = load_json_config(args.config)
    model_config = config["rectified_flow"]
    train_config = config["train"]

    device = get_device()
    dataset = load_dataset()
    rectified_flow = RectifiedFlow.from_config(model_config)
    trainer = RectifiedFlowTrainer(
        rectified_flow, dataset=dataset, device=device, **train_config
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
