from lightning.pytorch.cli import LightningCLI

from inpaint._runner import RunnerLightning
from inpaint.data import Data

def main():
    LightningCLI(
        model_class=RunnerLightning,
        datamodule_class=Data,

    )


if __name__ == '__main__':
    main()
