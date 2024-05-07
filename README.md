# GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis

------
### Installation Dependencies:
1. Installing Anaconda and Python (our version == 3.8.10).
2. Creating the new environment for Groot and installing the requirements.
   ~~~
   conda create -n Groot python=3.8
   conda activate Groot
   pip install -r requirements.txt
   ~~~

------
### Pretrained Models
Downloading the pretrained models and please place them into `pretrain/`.

Pretrained models can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1JTxQvPA-nnhVzMTh5wwwUtMMCT-fQVPg).

### Generative Model
As the paper described, we provide [Diffwave](https://github.com/lmnt-com/diffwave) as the diffusion model.

The pretrained model of DiffWave can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1JTxQvPA-nnhVzMTh5wwwUtMMCT-fQVPg) and please also place it into `pretrain/`.

We also provide the links for [WaveGrad](https://github.com/ivanvovk/WaveGrad) and [PriorGrad](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder).
