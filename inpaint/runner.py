from typing import Callable

import os
from dataclasses import dataclass
import numpy as np
import pickle

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
from wandb import Audio

# import distributed_util as dist_util
# from evaluation import build_resnet50

from . import util
# from .network import Image256Net
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
        config: DiffusionConfig,
        augment: AudioAugment,
        corrupt: Degradation,
        net: UNet,
    ) -> None:
        super().__init__()
        self.config = config
        self.augment = augment
        self.corrupt = corrupt
        self.net: torch.nn.Module = net 
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
        sched_dict = {'step_size': self.config.lr_step, 'gamma': self.config.lr_gamma}
        optimizer = torch.optim.AdamW(self.parameters(), **optim_dict)
        scheduler = lr_scheduler.StepLR(optimizer, **sched_dict)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def training_step(self, batch: Tensor):
        batch_size = batch.shape[0]
        y = self.augment(batch)
        x, mask = self.corrupt(y)

        step = torch.randint(0, self.config.interval, (batch_size,))
        xt = self.diffusion.q_sample(step, y, x, ot_ode=self.config.ot_ode)
        label = self.compute_label(step, y, xt)
        pred = self.net(xt, step)
        assert xt.shape == label.shape == pred.shape

        if mask is not None:
            pred = mask * pred
            label = mask * label

        loss = F.mse_loss(pred, label)

        return loss





# def build_optimizer_sched(opt, net, log):

#     optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
#     optimizer = AdamW(net.parameters(), **optim_dict)
#     log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

#     if opt.lr_gamma < 1.0:
#         sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
#         sched = lr_scheduler.StepLR(optimizer, **sched_dict)
#         log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
#     else:
#         sched = None


    # return optimizer, sched

# def make_beta_schedule(
#     n_timestep: int = 1000,
#     linear_start: float = 1e-4,
#     linear_end: float = 2e-2,
# ) -> np.ndarray:
#     # return np.linspace(linear_start, linear_end, n_timestep)
#     betas = (
#         torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
#     )
#     return betas.numpy()

# # def all_cat_cpu(opt, log, t):
# #     if not opt.distributed: return t.detach().cpu()
# #     gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
# #     return torch.cat(gathered_t).detach().cpu()

# class Runner(object):
#     def __init__(self, opt, log):
#         super(Runner,self).__init__()


#         betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
#         betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
#         self.diffusion = Diffusion(betas, opt.device)
#         log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

#         noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
#         self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
#         self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)


#         self.net.to(opt.device)
#         self.ema.to(opt.device)

#         self.log = log

#     # def compute_label(self, step, x0, xt):
#     #     """ Eq 12 """
#     #     std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
#     #     label = (xt - x0) / std_fwd
#     #     return label.detach()

#     # def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
#     #     """ Given network output, recover x0. This should be the inverse of Eq 12 """
#     #     std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
#     #     pred_x0 = xt - std_fwd * net_out
#     #     if clip_denoise: pred_x0.clamp_(-1., 1.)
#     #     return pred_x0

#     def sample_batch(self, opt, loader, corrupt_method):
#         if opt.corrupt == "mixture":
#             clean_img, corrupt_img, y = next(loader)
#             mask = None
#         elif "inpaint" in opt.corrupt:
#             clean_img, y = next(loader)
#             with torch.no_grad():
#                 corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
#         else:
#             clean_img, y = next(loader)
#             with torch.no_grad():
#                 corrupt_img = corrupt_method(clean_img.to(opt.device))
#             mask = None

#         # os.makedirs(".debug", exist_ok=True)
#         # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
#         # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
#         # debug()

#         y  = y.detach().to(opt.device)
#         x0 = clean_img.detach().to(opt.device)
#         x1 = corrupt_img.detach().to(opt.device)
#         if mask is not None:
#             mask = mask.detach().to(opt.device)
#             x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
#         cond = x1.detach() if opt.cond_x1 else None

#         if opt.add_x1_noise: # only for decolor
#             x1 = x1 + torch.randn_like(x1)

#         assert x0.shape == x1.shape

#         return x0, x1, mask, y, cond

#     def train(self, opt, train_dataset, val_dataset, corrupt_method):
#         self.writer = util.build_log_writer(opt)
#         log = self.log

#         net = DDP(self.net, device_ids=[opt.device])
#         ema = self.ema
#         optimizer, sched = build_optimizer_sched(opt, net, log)

#         train_loader = util.setup_loader(train_dataset, opt.microbatch)
#         val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

#         self.accuracy = torchmetrics.Accuracy().to(opt.device)
#         self.resnet = build_resnet50().to(opt.device)

#         net.train()
#         n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
#         for it in range(opt.num_itr):
#             optimizer.zero_grad()

#             for _ in range(n_inner_loop):
#                 # ===== sample boundary pair =====
#                 x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

#                 # ===== compute loss =====
#                 step = torch.randint(0, opt.interval, (x0.shape[0],))

#                 xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
#                 label = self.compute_label(step, x0, xt)

#                 pred = net(xt, step, cond=cond)
#                 assert xt.shape == label.shape == pred.shape

#                 if mask is not None:
#                     pred = mask * pred
#                     label = mask * label

#                 loss = F.mse_loss(pred, label)
#                 loss.backward()

#             optimizer.step()
#             ema.update()
#             if sched is not None: sched.step()

#             # -------- logging --------
#             log.info("train_it {}/{} | lr:{} | loss:{}".format(
#                 1+it,
#                 opt.num_itr,
#                 "{:.2e}".format(optimizer.param_groups[0]['lr']),
#                 "{:+.4f}".format(loss.item()),
#             ))
#             if it % 10 == 0:
#                 self.writer.add_scalar(it, 'loss', loss.detach())

#             if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
#                 net.eval()
#                 self.evaluation(opt, it, val_loader, corrupt_method)
#                 net.train()
#         self.writer.close()

#     @torch.no_grad()
#     def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

#         # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
#         # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
#         # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
#         nfe = nfe or opt.interval-1
#         assert 0 < nfe < opt.interval == len(self.diffusion.betas)
#         steps = util.space_indices(opt.interval, nfe+1)

#         # create log steps
#         log_count = min(len(steps)-1, log_count)
#         log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
#         assert log_steps[0] == 0
#         self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

#         x1 = x1.to(opt.device)
#         if cond is not None: cond = cond.to(opt.device)
#         if mask is not None:
#             mask = mask.to(opt.device)
#             x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

#         with self.ema.average_parameters():
#             self.net.eval()

#             def pred_x0_fn(xt, step):
#                 step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
#                 out = self.net(xt, step, cond=cond)
#                 return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

#             xs, pred_x0 = self.diffusion.ddpm_sampling(
#                 steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
#             )

#         b, *xdim = x1.shape
#         assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

#         return xs, pred_x0

#     @torch.no_grad()
#     def evaluation(self, opt, it, val_loader, corrupt_method):

#         log = self.log
#         log.info(f"========== Evaluation started: iter={it} ==========")

#         img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

#         x1 = img_corrupt.to(opt.device)

#         xs, pred_x0s = self.ddpm_sampling(
#             opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
#         )

#         log.info("Collecting tensors ...")
#         img_clean   = all_cat_cpu(opt, log, img_clean)
#         img_corrupt = all_cat_cpu(opt, log, img_corrupt)
#         y           = all_cat_cpu(opt, log, y)
#         xs          = all_cat_cpu(opt, log, xs)
#         pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

#         batch, len_t, *xdim = xs.shape
#         assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
#         assert xs.shape == pred_x0s.shape
#         assert y.shape == (batch,)
#         log.info(f"Generated recon trajectories: size={xs.shape}")

#         def log_image(tag, img, nrow=10):
#             self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

#         def log_accuracy(tag, img):
#             pred = self.resnet(img.to(opt.device)) # input range [-1,1]
#             accu = self.accuracy(pred, y.to(opt.device))
#             self.writer.add_scalar(it, tag, accu)

#         log.info("Logging images ...")
#         img_recon = xs[:, 0, ...]
#         log_image("image/clean",   img_clean)
#         log_image("image/corrupt", img_corrupt)
#         log_image("image/recon",   img_recon)
#         log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
#         log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

#         log.info("Logging accuracies ...")
#         log_accuracy("accuracy/clean",   img_clean)
#         log_accuracy("accuracy/corrupt", img_corrupt)
#         log_accuracy("accuracy/recon",   img_recon)

#         log.info(f"========== Evaluation finished: iter={it} ==========")
#         torch.cuda.empty_cache()