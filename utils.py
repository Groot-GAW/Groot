import numpy as np
import torch
import torch.nn as nn
import torchaudio
import math
from dataset import transform
from torch.autograd import Variable
import torch.nn.functional as F


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


class MCD(nn.Module):
    """ @pymcd """
    def __init__(self, sr=22050):
        super().__init__()

        self.sr = sr

    def logSpec_dist(self, x, y):
        log_spec_db_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y

        return log_spec_db_const * math.sqrt(np.inner(diff, diff))

    def forward(self, est, ori):
        est_mel = transform(est, self.sr)
        ori_mel = transform(ori, self.sr)

        est_inp = est_mel[-1, :]
        ori_inp = ori_mel[-1, :]

        mcd = self.logSpec_dist(est_inp, ori_inp)

        return mcd


class SSIM(nn.Module):
    """ @https://github.com/keonlee9420/DiffGAN-TTS/blob/main/utils/tools.py """
    def __init__(self, sr=22050):
        super().__init__()

        self.sr = sr

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1)

    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window

        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def forward(self, inp, tar):
        inp_mel = transform(inp, self.sr).unsqueeze(0)
        tar_mel = transform(tar, self.sr).unsqueeze(0)

        ssim = self.ssim(inp_mel.unsqueeze(0), tar_mel.unsqueeze(0))

        return ssim
