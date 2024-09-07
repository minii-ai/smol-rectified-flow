import torch
import torch.nn as nn
import torch.nn.functional as F
from improved_diffusion.unet import UNetModel
from tqdm import tqdm
from typing import Tuple


def extend(t: torch.Tensor, shape: torch.Size):
    return t.view(-1, *(1 for i in range(len(shape) - 1)))


class RectifiedFlow(nn.Module):
    def __init__(self, model: UNetModel):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def from_config(config: dict):
        unet = UNetModel(**config)
        return RectifiedFlow(unet)

    @staticmethod
    def add_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        t = extend(t, x0.shape)
        xt = t * noise + (1 - t) * x0
        return xt

    @staticmethod
    def get_timesteps(num_steps: int):
        timesteps = torch.linspace(1, 0, num_steps + 1)
        return timesteps.tolist()[:-1]

    @torch.inference_mode()
    def sample(
        self,
        num_steps: int = 16,
        num_samples: int = 1,
        shape: Tuple[int, ...] = None,
        cond: torch.Tensor = None,
    ):
        assert (
            shape is not None
        ), "Pass a shape that is not None to sample with rectified flow"

        dt = 1 / num_steps
        x = torch.randn((num_samples, *shape), device=self.device)
        timesteps = RectifiedFlow.get_timesteps(num_steps)

        # euler method
        for t in tqdm(timesteps):
            t = torch.tensor(
                [t for _ in range(num_samples)],
                device=self.device,
            )
            pred = self.model(x, t, cond)
            x = x - dt * pred

        return x

    def forward(self, x0: torch.Tensor, cond: torch.Tensor = None):
        B = x0.shape[0]

        # sample time step uniformly between [0, 1]
        t = torch.rand((B,), device=x0.device)

        # add noise to x0 to get xt
        noise = torch.randn_like(x0, device=x0.device)
        xt = RectifiedFlow.add_noise(x0, t, noise)

        # get drift x1 - x0 and predict it
        drift = noise - x0
        pred_drift = self.model(xt, t, cond)
        loss = F.mse_loss(pred_drift, drift)

        return loss
