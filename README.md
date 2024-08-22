# SCONE
This repository serves as the official code release of the AAAI24 paper: **Learning Spatially Collaged Fourier Bases for Implicit Neural Representation**
![](assets/logo.png?v=1&type=image)
<div align="center">
    <a href="https://arxiv.org/abs/2312.17018"><img src="https://img.shields.io/badge/Arxiv-2312.17018-b31b1b.svg?logo=arXiv" alt=""></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt=""></a>
</div>
<br>
<div align="center">
<strong><a href="https://www.linkedin.com/in/jason-chun-lok-li-0590b3166/"><u>Jason Chun Lok Li</u></a><sup>*</sup></strong>, <strong><a href="https://github.com/Cliu2"><u>Chang Liu</u></a><sup>*</sup></strong>, Binxiao Huang,  Ngai Wong
</div>
<br>
<div align="center">
<strong><sup>*</sup>Contributed Equally</strong>
</div>
<br>
<div align="center">
Department of Electrical and Electronic Engineering, The University of Hong Kong
</div>
<div align="center">
</div>

## ⚙️ Dependency
```
conda create -n scone python=3.9
conda activate scone
pip install -r requirements.txt
```

## 🏗️ Code Structure

The repository contains training scripts `train_<image/video/sdf>.py` for various data modalities (image, video, SDF) as described in our paper. For convenience, we provide bash scripts in the `scripts/` directory for quick start. Configuration files, including model and experiment settings, are stored as `.yaml` files under the `config/` directory.

## 🧪 Experiments
### Image
The Kodak dataset can be downloaded from [this link](https://r0k.us/graphics/kodak/). After downloading, please place the dataset in the `data/kodak` directory. To select which model to experiment, you can modify the `model_config` argument in the `train_image.sh` script. To train the model on all Kodak images in a single run, execute the following command in your terminal: 

```bash
./scripts/train_image.sh
```

### Video
The original cat video is available [here](https://www.pexels.com/video/the-full-facial-features-of-a-pet-cat-3040808). We have prepared for you the downsampled `cat.npy` file, which can be found in [this link](https://connecthkuhk-my.sharepoint.com/personal/jasonlcl_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjasonlcl%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fcat%2Enpy&parent=%2Fpersonal%2Fjasonlcl%5Fconnect%5Fhku%5Fhk%2FDocuments&ga=1). Place it under the `data/` folder. Once the data is ready, you can train the model on the cat video by executing the following command in your terminal:

```bash
./scripts/train_video.sh
```

### SDF 
The Stanford 3D scan dataset is available [here](https://graphics.stanford.edu/data/3Dscanrep/). Download the `.xyz` files and place them in the `data/stanford3d/` directory. Then, execute the command to start training on SDF data:

```bash
./scripts/train_sdf.sh
```

## 📝Citation

If you find SCONE is useful for your research and applications, consider citing it with the following BibTeX:
```
@inproceedings{li2024learning,
  title={Learning Spatially Collaged Fourier Bases for Implicit Neural Representation},
  author={Li, Jason Chun Lok and Liu, Chang and Huang, Binxiao and Wong, Ngai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={12},
  pages={13492--13499},
  year={2024}
}

```


## 🙏🏼Acknowledgements
We have adapted some of our code from [COIN++](https://github.com/EmilienDupont/coinpp) and [BACON](https://github.com/computational-imaging/bacon). We sincerely thank them for their contributions to open source.