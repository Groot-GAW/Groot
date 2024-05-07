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
