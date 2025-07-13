# Reimplementation_of_StyleGAN-NADA
StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators (re-implementation)
* Reimplementattion of [paper](https://arxiv.org/abs/2108.00946) - "StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators".
* Link to [official implementation](https://github.com/rinongal/StyleGAN-nada).
## Task description
In this project, the main goal is to reimplement the StyleGAN-NADA paper (https://stylegan-nada.github.io/), which focuses on domain adaptation of GAN.
* The key objective is to learn how to generate images from a specific domain based on a given text prompt.
> This is an educational research project aimed at studying an actual scientific paper, reproducing its results, and analyzing its limitations.

**StyleGAN-NADA** is a method for adapting a GAN (in example, [Style GAN 2 in TensorFlow](https://github.com/NVlabs/stylegan2), [Style GAN 2 in PyTorch](https://github.com/rosinality/stylegan2-pytorch)) to new domains without the need for annotated data. 

The core idea is to keep one generator instance fixed while training the other, ensuring that the 'direction' between generated images in the latent space aligns with a given text-guided direction using the [CLIP](https://github.com/openai/CLIP).

After studying the paper, we implemented the training and inference code for this model. The resulting trained model can generate images from text-specified domains - for example, in anime style or as sketch drawings, or even transform any character into Shrek.

## Getting Started
**Prerequisites**
 - python 3.11 with installed libs: os, torch, torchvision, numpy, matplotlib, re
 - in jupiter notebook will be installed (requred modules): Ninja, ftfy, regex, tqdm
 - [Installed CLIP](https://github.com/openai/CLIP) (downloads by jupiter notebook automatically).
 - [Style GAN 2 (in Pytorch)](https://github.com/rosinality/stylegan2-pytorch) (downloads by jupiter notebook automatically).
 - [Weights for StyleGAN2-ffhq One-Shot Adaptation of GAN in Just One CLIP (pytorch)](https://huggingface.co/akhaliq/OneshotCLIP-stylegan2-ffhq/resolve/main/stylegan2-ffhq-config-f.pt) (downloads by jupiter notebook automatically).
 - [tools.py](https://raw.githubusercontent.com/vital83/Reimplementation_of_StyleGAN-NADA/main/tools.py) (downloads by jupiter notebook automatically).

## Usage

At the moment there is only one possible option to run the work [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DCwbuYemn5Yc-3_5RQ7U3vpd3SbbDI78?usp=sharing)


Open in Colab and run:
 - train colab jupiter notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r2MRBtmxpUGJfJ0lzH9BCreUeRFCKLwD?usp=sharing)
 - validation colab jupiter notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TJONxEAv0nLkggNEgIgrbHKzRXsKm2cp?usp=sharing)


You can just click `Runtime > Run all`, no other actions required.

## Result examples
* 01 Anime painting from face
![Image](https://github.com/vital83/Reimplementation_of_StyleGAN-NADA/blob/main/01_anime_painting_from_face.png)

* 02 Shrek from face
![Image](https://github.com/vital83/Reimplementation_of_StyleGAN-NADA/blob/main/02_100_shrek_from_face.png)

* 03 Sketch from face
![Image](https://github.com/vital83/Reimplementation_of_StyleGAN-NADA/blob/main/04_10l_300s_sketch_from_face.png)

* 04 Zoombie from face
![Image](https://github.com/vital83/Reimplementation_of_StyleGAN-NADA/blob/main/05_5l_300s_zoombie_from_face.png)

* 05 Pixar cartoon from face
![Image](https://github.com/vital83/Reimplementation_of_StyleGAN-NADA/blob/main/03_300_pixar_from_face.png)


[Model weights (saved checkpoints)](https://drive.google.com/drive/folders/1a27Qx_Te2DQ95gjUhUEUW6K6VmFvcc4F?usp=sharing)


## Report

[Here you can find the work report](REPORT.md)
