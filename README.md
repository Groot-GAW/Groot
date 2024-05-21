# GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis

------
### :meta:Installation Dependencies:
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

### Generative Models
As the paper described, we provide [DiffWave](https://github.com/lmnt-com/diffwave) as the diffusion model.

The pretrained model of DiffWave can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1JTxQvPA-nnhVzMTh5wwwUtMMCT-fQVPg) and please also place it into `pretrain/`.

We also provide the links for [WaveGrad](https://github.com/ivanvovk/WaveGrad) and [PriorGrad](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder).

~~~
${Groot}
|-- diffwave
|-- pretrain        <-- the downloaded pretrained models
|-- inference.py
|-- model.py
|-- other python codes, config, LICENSE and README files
~~~

------
### Dataset:
The pretrained models correspond to LJspeech dataset. Here, we provide the link to download [LJspeech](https://keithito.com/LJ-Speech-Dataset/).

The LibriTTS and LibriSpeech datasets can be downloaded from [torchaudio](https://pytorch.org/audio/stable/datasets.html). That's exactly how we downloaded it.

------
### Inference
You can utilize pre-trained models to assess Groot's performance at 100 bps capacity using the LJSpeech dataset.
~~~
python inference.py --dataset_path path_to_your_test_dataset --encoder path_to_encoder --decoder path_to_decoder --diffwave path_to_generative_model
~~~

------
### License
This project is released under the MIT license. See [LICENSE](https://github.com/Groot-GAW/Groot/blob/main/LICENSE) for details.

------
### Citation



