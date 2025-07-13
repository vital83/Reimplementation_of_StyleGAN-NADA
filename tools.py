import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from model import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"


# Image tools
def clip_preprocess_tensor(imgs):
    imgs = (imgs.clamp(-1,1) + 1)/2
    imgs_224 = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, -1, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, -1, 1, 1)
    imgs_224 = (imgs_224 - clip_mean) / clip_std

    return imgs_224

# Tools
def directional_loss(img_features, text_features):
    """
    Вычисление направленного CLIP-лосса

    :param img_features: Визуальные эмбеддинги сгенерированных изображений
    :return: Значение лосса
    """
    # Нормализация эмбеддингов
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # Вычисление направлений
    text_direction = (text_features[1] - text_features[0]).unsqueeze(0)
    img_direction = img_features - text_features[0]

    # Косинусное расстояние
    cos_sim = torch.cosine_similarity(img_direction, text_direction, dim=-1)
    return 1 - cos_sim.mean()

# Layers processing tools

def get_all_conv_and_to_rgb_layers(g):
    """Получение всех слоев генератора из групп convs и to_rgbs"""
    layers = []
    # Собираем все сверточные и линейные слои
    for name, module in g.convs.named_modules():
        # if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            # Фильтруем слои только из сети синтеза
            # if 'synthesis' in name or 'conv' in name or 'to_rgb' in name:
            if 'ModulatedConv2d' in type(module).__name__:
                layers.append((name, module))

    for name, module in g.to_rgbs.named_modules():
        # if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            # Фильтруем слои только из сети синтеза
            # if 'synthesis' in name or 'conv' in name or 'to_rgb' in name:
            if 'ModulatedConv2d' in type(module).__name__:
                layers.append((name, module))

    return layers

def unfreeze_all_layers(g):
    """
    Размораживает все слои генератора

    :g: генератор
    """
    # Размораживаем все параметры
    for param in g.parameters():
        param.requires_grad = True

    return g

def freeze_all_except(g, layer_names=None):
    """
    Замораживает все параметры генератора, кроме указанных слоев

    :param layer_names: Список имен слоев для разморозки
    """
    # Замораживаем все параметры
    for param in g.parameters():
        param.requires_grad = False

    # Если не указаны слои - оставляем все замороженными
    if layer_names is None:
        return g

    # Размораживаем выбранные слои
    for name, param in g.named_parameters():
        # Проверяем все подмодули, содержащие имя слоя
        for layer_name in layer_names:
            if re.search(rf'\.{layer_name}(\.|$)', name):
                param.requires_grad = True
                break
    return g

def freeze_all_convs_except(g, layer_names=None):
    """
    Замораживает все convs параметры генератора, кроме указанных слоев

    :param layer_names: Список имен слоев для разморозки
    """
    # Замораживаем все convs параметры
    for param in g.convs.parameters():
        param.requires_grad = False

    # Если не указаны слои - оставляем все замороженными
    if layer_names is None:
        return g

    # Размораживаем выбранные слои
    for name, param in g.named_parameters():
        # Проверяем все подмодули, содержащие имя слоя
        for layer_name in layer_names:
            if (("convs." in name) and (re.search(rf'\.{layer_name}(\.|$)', name))):
                param.requires_grad = True
                break
    return g
