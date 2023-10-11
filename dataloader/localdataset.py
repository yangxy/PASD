import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torch.utils import data as data

from .realesrgan import RealESRGAN_degradation
from myutils.img_util import convert_image_to_fn, exists

class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="datasets/pngtxt", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio

        self.degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to, image_size) if exists(convert_image_to) else nn.Identity()
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.img_preproc = transforms.Compose([
            #transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_paths = []
        folders = os.listdir(pngtxt_dir)
        for folder in folders:
            self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/*.png'))[:])

    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        txt_path = img_path.replace(".png", ".txt")
        image = Image.open(img_path).convert('RGB')

        image = self.crop_preproc(image)
            
        example["pixel_values"] = self.img_preproc(image)
        if self.control_type is not None:
            if self.control_type == 'realisr':
                GT_image_t, LR_image_t = self.degradation.degrade_process(np.asarray(image)/255., resize_bak=self.resize_bak)
                example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
                example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0
            elif self.control_type == 'grayscale':
                image = np.asarray(image.convert('L').convert('RGB'))/255.
                example["conditioning_pixel_values"] = torch.from_numpy(image).permute(2,0,1)
            else:
                raise NotImplementedError

        fp = open(txt_path, "r")
        caption = fp.readlines()[0]
        if self.tokenizer is not None:
            example["input_ids"] = self.tokenize_caption(caption).squeeze(0)
        fp.close()

        return example

    def __len__(self):
        return len(self.img_paths)