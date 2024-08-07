# GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis

Official implementation for the paper "GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis" in ACM MM 2024.

*Weizhi Liu, Yue Li, Dongdong Lin, Hui Tian, Haizhou Li*. **China**.

Audio samples are available on our [Website](https://groot-gaw.github.io/).

------
![Static Badge](https://img.shields.io/badge/PYTHON-3.8%2B-blue)
![Static Badge](https://img.shields.io/badge/Groot-Generative_Watermarking-66c2a5?labelColor=e5f5f9)
### :loudspeaker: Installation Dependencies:
1. Installing Anaconda and Python (our version == 3.8.10).
2. Creating the new environment for Groot and installing the requirements.
   ~~~
   conda create -n Groot python=3.8
   conda activate Groot
   pip install -r requirements.txt
   ~~~

------
### Pretrained Models :link:
Downloading the pretrained models and please place them into :file_folder:`pretrain/`.

Pretrained models can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1JTxQvPA-nnhVzMTh5wwwUtMMCT-fQVPg).

### Generative Models :link:
As the paper described, we provide [DiffWave](https://github.com/lmnt-com/diffwave) as the diffusion model.

The pretrained model of DiffWave can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1JTxQvPA-nnhVzMTh5wwwUtMMCT-fQVPg) and please also place it into :file_folder:`pretrain/`.

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
### :notes: Dataset:
The pretrained models correspond to LJspeech dataset. Here, we provide the link to download [LJspeech](https://keithito.com/LJ-Speech-Dataset/).

The LibriTTS and LibriSpeech datasets can be downloaded from [torchaudio](https://pytorch.org/audio/stable/datasets.html).

------
### :rocket: Inference
You can utilize pre-trained models to assess Groot's performance at 100 bps capacity using the LJSpeech dataset.
~~~
python inference.py --dataset_path path_to_your_test_dataset \
                    --encoder path_to_encoder \
                    --decoder path_to_decoder \
                    --diffwave path_to_generative_model
~~~

------
### :heartpulse: Acknowledgement
[1] DiffWave: :newspaper:[[paper]](https://arxiv.org/pdf/2009.09761) :computer:[[code]](https://github.com/lmnt-com/diffwave)
[2] WaveGrad: :newspaper:[[paper]](https://arxiv.org/pdf/2009.00713) :computer:[[code]](https://github.com/ivanvovk/WaveGrad)
[3] PriorGrad: :newspaper:[[paper]](https://arxiv.org/pdf/2106.06406) :computer:[[code]](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder)

------
### :mortar_board:License
This project is released under the MIT license. See [LICENSE](https://github.com/Groot-GAW/Groot/blob/main/LICENSE) for details.

------
### :book: Citation
If you find the code and dataset useful in your research, please consider citing our paper:
```
@inproceedings{liu2024groot,
  title={GROOT: Generating Robust Watermark for Diffusion-Model-Based Audio Synthesis},
  author={Liu, Weizhi and Li, Yue and Lin, Dongdong and Tian, Hui and Li, Haizhou},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```


