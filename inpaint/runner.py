from dataclasses import dataclass
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import lr_scheduler
# from torch.nn.parallel import DistributedDataParallel as DDP

# from torch_ema import ExponentialMovingAverage
# import torchvision.utils as tu
# import torchmetrics
# from wandb import Audio


from inpaint.diffusion import Diffusion
from inpaint.data import AudioAugment, Degradation
from inpaint.unet import UNet

# from ipdb import set_trace as debug

import lightning

@dataclass
class DiffusionConfig:
    lr: float
    lr_step: int = 100
    lr_gamma: float = 1.0  # alt: 0.1
    l2_norm: float = 0.0
    interval: int = 1000
    beta_max: float = 0.3
    t0: float = 1e-4
    t1: float = 1.0
    ot_ode: bool = False

    @property
    def beta_schedule(self):
        n_timestep = self.interval
        start = 1e-4
        end = self.beta_max / self.interval
        betas = (
            torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
        betas = betas.numpy()
        return np.concatenate([betas[:n_timestep//2], np.flip(betas[:n_timestep//2])])



class RunnerLightning(lightning.LightningModule):
    def __init__(
        self,
        net: UNet,
        augment: AudioAugment,
        corrupt: Degradation,
        config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.augment = augment
        self.corrupt = corrupt
        self.net = net 
        self.diffusion = Diffusion(self.config.beta_schedule, self.device)
    
    def compute_label(self, step: Tensor, x0: Tensor, xt: Tensor):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step: int, xt: Tensor, net_out: Tensor):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        return pred_x0

    
    def configure_optimizers(self):
        optim_dict = {'lr': self.config.lr, 'weight_decay': self.config.l2_norm}
        # sched_dict = {'step_size': self.config.lr_step, 'gamma': self.config.lr_gamma}
        optimizer = torch.optim.AdamW(self.parameters(), **optim_dict)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.3, total_iters=3)
        # scheduler = lr_scheduler.StepLR(optimizer, **sched_dict)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def training_step(self, batch: Tensor):
        batch_size = batch.shape[0]
        y = self.augment(batch)
        x, mask = self.corrupt(y)

        step = torch.randint(0, self.config.interval, (batch_size,), device=self.device)
        xt = self.diffusion.q_sample(step, y, x, ot_ode=self.config.ot_ode)
        label = self.compute_label(step, y, xt)
        pred = self.net(xt, step)
        assert xt.shape == label.shape == pred.shape

        if mask is not None:
            pred = mask * pred
            label = mask * label

        loss = F.mse_loss(pred, label)
        self.log('loss', loss, prog_bar=True)

        return loss
