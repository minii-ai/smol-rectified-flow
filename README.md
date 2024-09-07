# Rectified Flow

Implementation of "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
https://arxiv.org/abs/2209.03003 on MNIST

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python3 scripts/train_mnist.py -c ./configs/mnist_rectified_flow.json
```

Example config.json file

```json
{
  "rectified_flow": {
    "in_channels": 1,
    "model_channels": 32,
    "out_channels": 1,
    "num_res_blocks": 1,
    "attention_resolutions": [8],
    "dropout": 0,
    "channel_mult": [1, 2, 2, 2],
    "conv_resample": true,
    "dims": 2,
    "num_classes": 10,
    "use_checkpoint": false,
    "num_heads": 1,
    "num_heads_upsample": 1,
    "use_scale_shift_norm": false
  },
  "train": {
    "seed": 0,
    "lr": 3e-4,
    "num_train_steps": 10000,
    "save_dir": "./experiments/mnist",
    "checkpoint_every": 2500,
    "sample_every": 1000,
    "num_samples_per_class": 9
  }
}
```

## Inference

```bash
python3 script/inference_mnist.py -c ./configs/mnist_rectified_flow.py -w weights.pt -n 16 -k 1 -o ./output
```

- `-c` path to the config file
- `-w` path to the weights (.pt) file
- `-n` number of sampling steps
- `-k` class (0, 1, ... 9)
- `-o` output dir
