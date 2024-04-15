from dataclasses import dataclass
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
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
        self.lr = self.config.lr  # for automatic lr tuning
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
        optim_dict = {'lr': self.lr, 'weight_decay': self.config.l2_norm}
        # sched_dict = {'step_size': self.config.lr_step, 'gamma': self.config.lr_gamma}
        optimizer = torch.optim.AdamW(self.parameters(), **optim_dict)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.3, total_iters=3)
        # scheduler = lr_scheduler.StepLR(optimizer, **sched_dict)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def step_plots(self, x: Tensor, y: Tensor, xt: Tensor, label: Tensor, pred: Tensor, mask: Tensor, step: Tensor):
        # times = [0, 0.25, 0.5, 0.75, 1.0]
        batch_id = 0
        # steps = [int(t * self.config.interval) for t in times]
        mask_start = mask[batch_id].argmax()
        mask_end = mask_start + (~mask[batch_id, mask_start:]).argmax()
        diff = mask_end - mask_start
        start = mask_start - diff
        end = mask_end + diff

        plt.plot(xt[batch_id, start:end].detach().cpu(), linewidth=1, label='xt')
        plt.plot(y[batch_id, start:end].detach().cpu(), linewidth=1, label='y')
        plt.plot(label[batch_id, start:end].detach().cpu(), linewidth=1, label='label')
        plt.plot(pred[batch_id, start:end].detach().cpu(), linewidth=1, label='pred')
        plt.legend()
        plt.show()
    
    def plot_steps(self, batch: Tensor):
        from einops import repeat
        # times = [0, 0.25, 0.5, 0.75, 0.95]
        times = [0, 0.1, 0.2, 0.5, 0.75, 0.95]
        step = Tensor([int(self.config.interval * t) for t in times], device=batch.device).to(torch.int32)
        batch_id = 0
        y = batch
        x, mask = self.corrupt(y)
        y = y[batch_id]
        x = x[batch_id]
        mask = mask[batch_id]
        # print(x.shape)
        # print(y.shape)
        # print(mask.shape)
        x = repeat(x, 't -> b t', b=len(times))
        y = repeat(y, 't -> b t', b=len(times))
        mask = repeat(mask, 't -> b t', b=len(times))
        xt = self.diffusion.q_sample(step, y, x, ot_ode=self.config.ot_ode)
        xt = mask * xt + ~mask * y
        label = self.compute_label(step, y, xt)

        mask_start = (mask[batch_id] * 1).argmax()
        mask_end = mask_start + ((~mask[batch_id, mask_start:]) * 1).argmax()
        diff = mask_end - mask_start
        start = mask_start - diff
        end = mask_end + diff

        fig, axs = plt.subplots(nrows=len(times), ncols=1, figsize=(15, 20))
        for i, ax in enumerate(axs):
            # ax.plot(label[i, start:end].detach().cpu(), linewidth=1, label='label')
            ax.plot(xt[i, start:end].detach().cpu(), linewidth=1, label='xt')
            ax.plot(y[i, start:end].detach().cpu(), linewidth=1, label='y')
            ax.plot(x[i, start:end].detach().cpu(), linewidth=1, label='x')
            ax.set_title(f'Time = {times[i]}')
        fig.legend()
        fig.savefig(f'.beta-search/beta={self.config.beta_max:.4f}.pdf')


    
    def training_step(self, batch: Tensor):
        batch_size, samples = batch.shape
        y = self.augment(batch)
        x, mask = self.corrupt(y)

        step = torch.randint(0, self.config.interval, (batch_size,), device=self.device)
        xt = self.diffusion.q_sample(step, y, x, ot_ode=self.config.ot_ode)
        xt = mask * xt + ~mask * y  # inpainting task
        label = self.compute_label(step, y, xt)
        pred = self.net(xt, step)
        assert xt.shape == label.shape == pred.shape

        pred = mask * pred
        label = mask * label

        loss = F.mse_loss(pred, label)
        self.log('loss', loss, prog_bar=True)
        self.log('pred_std', ((pred - label).std() * samples / (mask * 1.0).sum()), prog_bar=True)

        return loss
