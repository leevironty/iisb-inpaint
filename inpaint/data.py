from pathlib import Path
from dataclasses import dataclass


import lightning
from pydantic import BaseModel, Field
import numpy.random as rng
from torch import Tensor
import torchaudio.functional as F
import torchaudio
from torch.utils.data import Dataset
import torch
from torch import Tensor

class Meta(BaseModel):
    file_path: str
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str

    @classmethod
    def from_path(cls, path: str | Path):
        meta = torchaudio.info(path)
        file_path = path if isinstance(path, str) else path.as_posix()
        return cls(
            file_path=file_path,
            sample_rate=meta.sample_rate,
            num_frames=meta.num_frames,
            num_channels=meta.num_channels,
            bits_per_sample=meta.bits_per_sample,
            encoding=meta.encoding,
        )


@dataclass
class AudioAugment:
    polarity: bool = True
    adjust_db_lb: float = -5
    adjust_db_ub: float = 5

    def __call__(self, wave: Tensor) -> Tensor:
        n_batch = wave.shape[0]
        mults = self.get_mult(n_batch)
        return wave * mults[:, None]

    def get_mult(self, n_batch) -> Tensor:
        polarity = torch.randint(0, 2, (n_batch,)) * 2 - 1
        r = torch.rand((n_batch,))
        db = self.adjust_db_lb + (self.adjust_db_ub - self.adjust_db_ub) * r
        amplitude = 10 ** db
        return polarity * amplitude


@dataclass
class Degradation:
    min_len: int
    avg_len: int
    max_len: int
    min_count: int
    max_count: int

    def __call__(self, wave: Tensor):
        batches, samples = wave.shape
        mask = torch.zeros_like(wave, dtype=torch.bool)
        n_blocks = torch.randint(
            low=self.min_count,
            high=self.max_count+1,
            size=(batches,)
        )
        for batch, blocks in enumerate(n_blocks):
            blocks = blocks.item()
            durations = -self.avg_len * (1 - torch.rand((blocks,))).log()
            durations = durations.clip(self.min_len, self.max_len).to(torch.int32)
            starts = torch.randint(0, samples, (blocks,))
            ends = starts + durations
            for start, end in zip(starts, ends):
                mask[batch, start:end] = True
        return wave * ~mask, mask

    #     # n_blocks = rng.randint(self.min_count, self.max_count)
    #     # for _ in range(n_blocks):
    #     #     duration = int(rng.exponential(scale=self.avg_len))
    #     #     duration = min(max(duration, self.min_len), self.max_len)
    #     #     start = rng.randint(0, samples)
    #     #     wv[start:start+duration] = 0
    #     # return wv 
    #     # return torch.cat([self.mask_one_sample(tensor) for tensor in wave]).detach()

    # def mask_one_sample(self, wv: Tensor) -> Tensor:
    #     wv = wv[:]
    #     samples = wv.shape[-1]
    #     n_blocks = rng.randint(self.min_count, self.max_count)
    #     for _ in range(n_blocks):
    #         duration = int(rng.exponential(scale=self.avg_len))
    #         duration = min(max(duration, self.min_len), self.max_len)
    #         start = rng.randint(0, samples)
    #         wv[start:start+duration] = 0
    #     return wv



class RandomTracks(Dataset):
    """Dataset of wav files with random sampling of position."""
    def __init__(
        self,
        root: str | Path,
        samples: int,
    ) -> None:

        print('Finding files...')
        self.samples_per_item = samples
        self.root = root if isinstance(root, Path) else Path(root)
        all_wavs = sorted(self.root.glob('**/*.wav'))
        print('Loading metadata...')
        self.meta = [Meta.from_path(path) for path in all_wavs]
        self.total_duration_seconds = sum(
            meta.num_frames / meta.sample_rate * meta.num_channels
            for meta in self.meta
        )
        
        print(f'Dataset init done! ({len(self.meta)} tracks)')
    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index: int) -> Tensor:
        meta = self.meta[index]
        # sample position, channel
        channel = rng.randint(0, meta.num_channels)
        start: int = rng.randint(0, meta.num_frames - self.samples_per_item)
        # load waveform
        sample = torchaudio.load(
            uri=meta.file_path,
            frame_offset=start,
            num_frames=self.samples_per_item,
            normalize=True,
        )[0][channel]  # load return waveform and sample rate
        if ((sample == 0) * 1.0).mean() > 0.2:
            return self[index]  # try again if we sample an empty section
        assert isinstance(sample, Tensor)
        return sample


class Data(lightning.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
