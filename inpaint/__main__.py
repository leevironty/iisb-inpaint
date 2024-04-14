from lightning.pytorch.cli import LightningCLI

from inpaint.runner import RunnerLightning
from inpaint.data import TrackDataModule

def main():
    LightningCLI(
        model_class=RunnerLightning,
        datamodule_class=TrackDataModule,
        # overwrites versioned config in tensorboard logging folder
        save_config_kwargs={"overwrite": True},
    )


if __name__ == '__main__':
    main()
