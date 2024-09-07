import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.adam import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from io import BytesIO


from .utils import get_device, count_parameters
from .rectified_flow import RectifiedFlow


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class RectifiedFlowTrainer:
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        dataset: Dataset,
        lr: float = 3e-4,
        batch_size: int = 16,
        num_train_steps: int = 50000,
        device: str = None,
        save_dir: str = None,
        checkpoint_every: int = 1000,
        sample_every: int = 1000,
        num_samples_per_class: int = 4,
        seed: int = 0,
    ):
        self.rectified_flow = rectified_flow
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = Adam(rectified_flow.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.device = device if device else get_device()
        self.save_dir = save_dir
        self.checkpoint_every = checkpoint_every
        self.sample_every = sample_every
        self.num_samples_per_class = num_samples_per_class
        self.seed = seed
        self.writer = SummaryWriter(log_dir=save_dir)

    def checkpoint(self, rectified_flow: RectifiedFlow, iteration: int):
        if iteration == self.num_train_steps:
            path = os.path.join(self.save_dir, "weights.pt")
        else:
            path = os.path.join(
                self.save_dir, "checkpoints", f"checkpoint_{iteration}.pt"
            )
        torch.save(rectified_flow.state_dict(), path)

    @torch.no_grad()
    def sample(self, rectified_flow: RectifiedFlow, iteration: int):
        rectified_flow.eval()

        num_classes = self.rectified_flow.model.num_classes
        samples_per_class = []
        for class_idx in range(num_classes):
            samples = rectified_flow.sample(
                num_steps=16,
                num_samples=self.num_samples_per_class,
                shape=(1, 32, 32),
                cond=torch.tensor(
                    [class_idx for _ in range(self.num_samples_per_class)],
                    device=self.device,
                ),
            )
            samples_per_class.append(samples)

        fig, axs = plt.subplots(num_classes, 1, figsize=(5, num_classes), squeeze=True)
        for class_idx, samples in enumerate(samples_per_class):
            img_grid = vutils.make_grid(
                samples, nrow=self.num_samples_per_class, pad_value=1
            )
            axs[class_idx].imshow(img_grid.permute(1, 2, 0).cpu(), cmap="gray")
            axs[class_idx].axis("off")
            axs[class_idx].set_title(f"{class_idx}")
            axs[class_idx].set_aspect("equal")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

        # Save figure to a buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert buffer image to tensor
        image_tensor = torch.tensor(plt.imread(buf)).permute(2, 0, 1)
        self.writer.add_image(f"samples", image_tensor, iteration)

    def train(self):
        torch.manual_seed(self.seed)
        rectified_flow = self.rectified_flow
        rectified_flow.to(self.device)
        rectified_flow.train()
        num_params = count_parameters(rectified_flow)

        print("[INFO] Training Rectified Flow")
        print(f"[INFO] Num Parameters: {num_params}")

        # create checkpoint dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

        train_loader = cycle(self.dataloader)
        pbar = tqdm(range(self.num_train_steps))
        for i in pbar:
            img, cond = next(train_loader)
            img, cond = img.to(self.device), cond.to(self.device)
            loss = rectified_flow(img, cond)

            pbar.set_postfix(loss=loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log loss
            self.writer.add_scalar("train/loss", loss.item(), global_step=i)

            if (i + 1) % self.checkpoint_every == 0:
                self.checkpoint(rectified_flow, i + 1)

            if (i + 1) % self.sample_every == 0:
                self.sample(rectified_flow, i + 1)
                rectified_flow.train()

        # save final checkpoint
        self.checkpoint(rectified_flow, self.num_train_steps)
        print("[INFO] Finished training")
