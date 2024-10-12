import os
import PIL
import cv2
import math
import numpy as np
import torch
import torchvision
import imageio

from einops import rearrange

def save_videos_grid(videos, path=None, rescale=True, n_rows=4, fps=8, discardN=0):
    videos = rearrange(videos, "b c t h w -> t b c h w").cpu()
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x / 2.0 + 0.5).clamp(0, 1)  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        #x = adjust_gamma(x, 0.5)
        outputs.append(x)

    outputs = outputs[discardN:]

    if path is not None:
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, outputs, duration=1000/fps, loop=0)

    return outputs

def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image

def colorful_loss(pred):
    colorfulness_loss = 0
    for i in range(pred.shape[0]):
        (R, G, B) = pred[i][0], pred[i][1], pred[i][2]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R+G) - B)
        (rbMean, rbStd) = (torch.mean(rg), torch.std(rg))
        (ybMean, ybStd) = (torch.mean(yb), torch.std(yb))
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = stdRoot + (0.3 * meanRoot)
        colorfulness_loss += (1 - colorfulness)
    return colorfulness_loss