import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as tf
from audioseal_attack import AudioEffects
from typing import Optional, Tuple
import math
import torch.nn.functional as F


class Attacker(nn.Module):
    def __init__(self, sr=22050):
        super().__init__()

        self.distortion = AudioEffects()
        self.sr = sr
        self.spr_factor = int(self.sr/2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # this function is from torchaudio
    def speed(
            self, waveform: torch.Tensor, orig_freq: int, factor: float, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        source_sample_rate = int(factor * orig_freq)
        target_sample_rate = int(orig_freq)

        gcd = math.gcd(source_sample_rate, target_sample_rate)
        source_sample_rate = source_sample_rate // gcd
        target_sample_rate = target_sample_rate // gcd

        if lengths is None:
            out_lengths = None
        else:
            out_lengths = torch.ceil(lengths * target_sample_rate / source_sample_rate).to(lengths.dtype)

        return torchaudio.functional.resample(waveform, source_sample_rate, target_sample_rate), out_lengths

    # this function is from torchaudio
    def add_noise(
            self, waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
            raise ValueError("Input leading dimensions don't match.")

        L = waveform.size(-1)

        if L != noise.size(-1):
            raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

        # compute scale
        if lengths is not None:
            mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
                -1
            )  # (*, L) < (*, 1) = (*, L)
            masked_waveform = waveform * mask
            masked_noise = noise * mask
        else:
            masked_waveform = waveform
            masked_noise = noise

        energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
        energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
        original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
        scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

        # scale noise
        scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

        return waveform + scaled_noise  # (*, L)

    def gaussian_noise(self, wmd, level=5.0):
        noise = torch.randn_like(wmd).to(self.device)
        noise_level = torch.tensor([level]).to(self.device)
        attacked = self.add_noise(wmd, noise, noise_level).unsqueeze(1)

        return attacked

    def lowpass_filtering(self, wmd):
        attacked = tf.lowpass_biquad(wmd, self.sr, 3000).unsqueeze(1)

        return attacked

    def bandpass_filtering(self, wmd):
        attacked = self.distortion.bandpass_filter(wmd.unsqueeze(1))

        return attacked

    def stretch(self, wmd):
        stretch_input, _ = self.speed(wmd.unsqueeze(1), self.sr, 0.5)
        attacked = F.interpolate(stretch_input, 22272, mode="linear")

        return attacked

    def suppress(self, wmd, front=True):
        crop_input = wmd.clone().unsqueeze(1)
        if front:
            crop_input[:, :, :self.spr_factor] = 0
            attacked = crop_input
        else:
            crop_input[:, :, self.spr_factor:] = 0
            attacked = crop_input

        return attacked

    def echo(self, wmd):
        attacked = self.distortion.echo(wmd.unsqueeze(1))

        return attacked

    def attack_function(self, wmd, choice=1, flag=True):
        if choice == 1:
            return self.gaussian_noise(wmd)
        elif choice == 2:
            return self.lowpass_filtering(wmd)
        elif choice == 3:
            return self.bandpass_filtering(wmd)
        elif choice == 4:
            return self.stretch(wmd)
        elif choice == 5:
            return self.suppress(wmd, front=flag)
        elif choice == 6:
            return self.echo(wmd)
        else:
            return wmd.unsqueeze(1)

    def forward(self, stego, choice=0, flag=True):
        attacked = self.attack_function(stego, choice, flag)

        return attacked





