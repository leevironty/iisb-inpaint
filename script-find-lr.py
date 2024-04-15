from lightning import Trainer
from inpaint.runner import DiffusionConfig, RunnerLightning
from inpaint.data import AudioAugment, Degradation, TrackDataModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateFinder
from omegaconf import OmegaConf
from jsonargparse import ArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
import torch
import numpy as np

from inpaint.unet import UNet


# parser = ArgumentParser()
# config = parser.parse_path('config.yaml')


config = OmegaConf.load('config.yaml')



betas_min = 0.001
betas_max = 0.3
n_evals = 20

betas = np.linspace(np.log(betas_min), np.log(betas_max), num=n_evals)
betas = np.exp(betas)

for beta in betas:
    diff_conf_args = {**config.model.config} | {'beta_max': beta}
    datamodule = TrackDataModule(**config.data)
    runner = RunnerLightning(
        net=UNet(**config.model.net.init_args),
        augment=AudioAugment(),
        corrupt=Degradation(**config.model.corrupt),
        config=DiffusionConfig(**diff_conf_args),
    )

    trainer = Trainer(min_epochs=1)
    tuner = Tuner(trainer)

    with torch.no_grad():
        runner.eval()
        datamodule.setup('fit')
        batch = datamodule.train_dataloader().__iter__().__next__()
        runner.plot_steps(batch)
        continue


    lr_finder = tuner.lr_find(
        model=runner,
        datamodule=datamodule,
        max_lr=0.01
    )
    assert lr_finder is not None

    fig = lr_finder.plot(suggest=True)
    # fig.savefig('')
    fig.savefig(f'.lr-find/beta={beta:.4f}.pdf')
    print(f'Suggested (beta={beta:.4f}): {lr_finder.suggestion()}')
