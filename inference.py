import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm

from dataset import LJSpeech
from torch.utils.data import DataLoader
from diffwave.model import DiffWave
from diffwave.params import AttrDict, params as base_params
import model
from utils import *
from pystoi import stoi
from torch_pesq import PesqLoss
from Attacks import Attacker

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def infer_for_diffwave(arg, config):
    set_seed(config["seed"])
    print(f"Inferring Groot!")

    """ parameters """
    watermark_length = config["watermark_length"]
    sr = config["sr"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    """ dataset """
    dataset = LJSpeech(root=arg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    """ model """
    diffwave_dir = arg.diffwave
    checkpoint = torch.load(diffwave_dir)
    diffwave = DiffWave(AttrDict(base_params)).to(device)
    diffwave.load_state_dict(checkpoint['model'])
    diffwave.eval()
    for dw_params in diffwave.parameters():
        dw_params.requires_grad = False
    encoder = model.WMEncoder(watermark_length=watermark_length).to(device)
    encoder.load_state_dict(torch.load(arg.encoder))
    encoder.eval()
    for dw_params in encoder.parameters():
        dw_params.requires_grad = False
    decoder = model.WMDecoder(watermark_length=watermark_length).to(device)
    decoder.load_state_dict(torch.load(arg.decoder))
    decoder.eval()
    for dw_params in decoder.parameters():
        dw_params.requires_grad = False
    
    """ inferring """
    epoch = 0
    mos_func = PesqLoss(0.5, sample_rate=sr)
    watermark = generate_secret(watermark_length=watermark_length, batch_size=batch_size).to(device)
    dw_args = cal_param()
    attacker = Attacker()
    iter = tqdm(dataloader)
    
    for mel, _ in iter:
        epoch += 1
        mel = mel.to(device)
        
        # watermarking
        latent_sigma = encoder(watermark)

        # generating
        audio, watermarked = audio_sample(diffwave, latent_sigma, mel, device, dw_args)

        # attack
        attacked = attacker(watermarked, choice=0, flag=True)

        # extracting
        extracted_watermark = decoder(attacked)

        # evaluation
        aud_numpy = audio[-1, :].cpu().numpy()
        wmd = attacked.squeeze(1)
        wmd_numpy = wmd[-1, :].cpu().numpy()

        stoi_score = stoi(aud_numpy, wmd_numpy, sr)
        mos_score = mos_func.mos(audio.cpu(), wmd.cpu())

        acc = cal_acc(watermark, extracted_watermark)

        log = f"=Epoch {epoch}= STOI: {stoi_score.item():.4f}, MOSL: {mos_score.item():.4f}, ACC: {acc.item():.4f}"
        iter.set_description(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to testing dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config")
    parser.add_argument("--encoder", default='pretrain/groot_enc_ljs100.pt', help='path to encoder')
    parser.add_argument("--decoder", default='pretrain/groot_dec_ljs100.pt', help='path to decoder')
    parser.add_argument("--diffwave", default="pretrain/diffwave.pt", help='path to diffwave')

    args = parser.parse_args()
    configs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    infer_for_diffwave(args, configs)
