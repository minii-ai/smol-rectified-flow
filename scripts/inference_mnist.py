import argparse
import json
import torch
import os
from glob import iglob
import re
import torchvision

import sys

import torchvision.transforms.functional

sys.path.append("./")

from rectified_flow import RectifiedFlow
from rectified_flow.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument("-w", "--weights", type=str, required=True)
    parser.add_argument("-n", "--num_steps", type=int, default=16)
    parser.add_argument("-k", "--cls", type=int, default=0)
    parser.add_argument("-o", "--output_dir", type=str, default="./output")

    return parser.parse_args()


def load_json_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def main(args):
    config = load_json_config(args.config)
    model_config = config["rectified_flow"]

    device = get_device()
    rectified_flow = RectifiedFlow.from_config(model_config)
    rectified_flow.load_state_dict(torch.load(args.weights, weights_only=True))
    rectified_flow.to(device)

    sample = rectified_flow.sample(
        num_samples=1,
        num_steps=args.num_steps,
        shape=(1, 32, 32),
        cond=torch.tensor([args.cls], device=device),
    )

    # convert tensor to image
    sample = torch.clamp(sample, 0, 1)
    image = torchvision.transforms.functional.to_pil_image(sample[0].cpu(), mode="L")

    # https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py#L146
    output_name = os.path.join(args.output_dir, "img_{idx}.jpg")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        idx = 0
    else:
        fns = [
            fn
            for fn in iglob(output_name.format(idx="*"))
            if re.search(r"img_[0-9]+\.jpg$", fn)
        ]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    save_path = output_name.format(idx=idx)
    image.save(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
