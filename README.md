# Reimplementation_of_StyleGAN-NADA
StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators (re-implementation)
* Реимплементация [статьи](https://arxiv.org/abs/2108.00946) - "StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators".
* Ссылка на [официальную имплементацию](https://github.com/rinongal/StyleGAN-nada).
## Описание задачи
В данном проекте основной задачей является реимплементация статьи StyleGAN-NADA (https://stylegan-nada.github.io/) - задача domain адаптации предобученного гана.
* Основная задача - научиться генерировать картинки из определенного домена на основе предложенного текстового промпта. 
> Это учебный исследовательский проект, направленный на изучение настоящей научной статьи, воспроизведение результатов и анализ недостатков.

**StyleGAN-NADA** — это метод адаптации GAN (например, [Style GAN 2 in TensorFlow](https://github.com/NVlabs/stylegan2), [Style GAN 2 in PyTorch](https://github.com/rosinality/stylegan2-pytorch)) к новым доменам без необходимости аннотированных данных. 

Основная идея состоит в том, чтобы сохранить один экземпляр генератора постоянным, а другой обучить так, чтобы «направление» между сгенерированными изображениями в векторном пространстве совпадало с заданным текстовым направлением при помощи модели [CLIP](https://github.com/openai/CLIP).

После изучения статьи написан код для обучения и инференса такой модели. В итоге обученная модель умеет генерировать изображения из заданного текстовым промптом домена, например, в стиле аниме или скетч картинки, или превращать любого персонажа в Шрека. Дополнительно с помощью Streamlit сделан веб-сервис для использования такой модели.

## Getting Started
**Prerequisites**
  - [Installed CLIP](https://github.com/openai/CLIP)
  - [Style GAN 2 in Pytorch)](https://github.com/rosinality/stylegan2-pytorch)
  - [Weights for StyleGAN2-ffhq One-Shot Adaptation of GAN in Just One CLIP (pytorch)](https://huggingface.co/akhaliq/OneshotCLIP-stylegan2-ffhq/resolve/main/stylegan2-ffhq-config-f.pt)
 
## Usage

At the moment there is only one possible option to run the work [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DCwbuYemn5Yc-3_5RQ7U3vpd3SbbDI78?usp=sharing)
You can just click `Runtime > Run all`, no other actions required.

## Report

[Here you can find the work report](REPORT.md)
