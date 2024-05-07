import os
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as TT
from torch.utils.data import Dataset


def transform(wav, sr, hop=256):
    speech = torch.clamp(wav[0], -1.0, 1.0)

    if speech.device.type == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    mel_args = {
        'sample_rate': sr,
        'win_length': hop * 4,
        'hop_length': hop,
        'n_fft': hop * 4,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': 80,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args).to(device)

    with torch.no_grad():
        spectrogram = mel_spec_transform(speech)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

    return spectrogram


class LJSpeech(Dataset):
    def __init__(self, root, spe_len=22050, s_len=1):
        self.root = root
        self.fixed = spe_len * s_len
        self.file_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.wav')]

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[1] < self.fixed:
            waveform = F.pad(waveform, (0, self.fixed - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.fixed]
        mel = transform(waveform, 22050)

        return mel, sample_rate

    def __len__(self):
        return len(self.file_paths)
