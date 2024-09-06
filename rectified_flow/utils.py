import torch


def get_device():
    # Check if MPS (Metal Performance Shaders) is available (for macOS with Apple Silicon)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Check if CUDA is available (for NVIDIA GPUs)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # Default to CPU
    else:
        return torch.device("cpu")


def denormalize_image(image: torch.Tensor):
    """
    Maps image with min and max value of [-1, 1] to [0, 1]
    """
    return torch.clamp((image + 1) * 0.5, 0, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
