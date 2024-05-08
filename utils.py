import numpy as np
import torch
import torch.nn as nn
import torchaudio
import math
from dataset import transform
from torch.autograd import Variable
import torch.nn.functional as F
from diffwave.params import AttrDict, params as base_params


def generate_secret(watermark_length, batch_size=1):
    secret = torch.randint(low=0, high=2, size=(batch_size, watermark_length)).float()
    return secret


def cal_acc(watermark, recover):
    sec_pred = (recover > 0).float()
    bitwise_acc = 1.0 - torch.mean(torch.abs(watermark - sec_pred))

    return bitwise_acc


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def cal_param(fast_sampling=True):
    training_noise_schedule = np.array(base_params.noise_schedule)
    inference_noise_schedule = np.array(
        base_params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                T.append(t + twiddle)
                break
    T = np.array(T, dtype=np.float32)

    return alpha, alpha_cum, beta, T


def audio_sample(diffwave, latent_sigma, mel, device, dw_args):
    alpha = dw_args[0]
    alpha_cum = dw_args[1]
    beta = dw_args[2]
    T = dw_args[3]

    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    mel = mel.to(device)

    audio = torch.randn(mel.shape[0], base_params.hop_samples * mel.shape[-1], device=device)
    watermarked = latent_sigma

    for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n] ** 0.5
        c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

        audio = c1 * (audio - c2 * diffwave(audio, mel, torch.tensor([T[n]], device=device)).squeeze(1))
        watermarked = c1 * (
                    watermarked - c2 * diffwave(watermarked, mel, torch.tensor([T[n]], device=device)).squeeze(1))

        if n > 0:
            noise_audio = torch.randn_like(audio)
            noise_wm = torch.randn_like(watermarked)
            sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
            audio += sigma * noise_audio
            watermarked += sigma * noise_wm

        audio = torch.clamp(audio, -1.0, 1.0)
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

    return audio, watermarked
