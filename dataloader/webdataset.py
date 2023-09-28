import cv2
import random
import torch
from torch import nn
from functools import partial
import webdataset as wds
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation
from myutils import convert_image_to_fn, exists

def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    for sample in samples:
        try:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
key_verifier = wds.filters.pipelinefilter(verify_keys)

def rename(filename):
    name = ''.join(random.sample('abcdefghigklmnopqrstuvwxyz', 5))
    return f'{name}_{filename}'

def tarfile_samples(src, handler=wds.handlers.reraise_exception, select_files=None, rename_files=rename):
    streams = wds.tariterators.url_opener(src, handler=handler)
    files = wds.tariterators.tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )   
    samples = wds.tariterators.group_by_keys(files, handler=handler)
    return samples

tarfile_to_samples = wds.filters.pipelinefilter(tarfile_samples)

class WebImageDataset(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(
            self,
            urls=None,
            image_size=512,
            tokenizer=None,
            accelerator=None,
            control_type=None,
            null_text_ratio=0.0,
            center_crop=False,
            random_flip=True,
            resize_bak=True,
            convert_image_to="RGB",
            extra_keys=[],
            handler=wds.handlers.warn_and_continue, #reraise_exception,
            resample=False,
            shuffle_shards=True
    ):
        super().__init__()
        keys = ["jpg", 'txt'] + extra_keys
        self.resampling = resample
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio

        self.degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
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
        
        if exists(urls):
            wds_urls = urls
        else:
            #urls = []
            #with open('datasets/tarlist.txt', 'r') as fp:
            #    urls = [line for line in fp.readlines()]
            wds_urls = "http://your_url_path/Laion-high-resolution/{000000..017535}.tar"
        
        # Add the shardList and randomize or resample if requested
        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(wds_urls))
        else:
            self.append(wds.SimpleShardList(wds_urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(10000))
        
        self.append(wds.tarfile_to_samples(handler=handler))
        #self.append(tarfile_to_samples(handler=handler)) # in case of duplicated filename
        self.append(wds.decode("pilrgb", handler=handler))
        
        self.append(key_verifier(required_keys=keys, handler=handler))

        # Apply preprocessing
        self.append(wds.map(self.preproc))

    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def preproc(self, sample):
        example = dict()

        """Applies the preprocessing for images"""
        if self.img_preproc is not None:
            image = self.crop_preproc(sample["jpg"])
            
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

        caption = sample['txt'] if 'txt' in sample else ''
        if self.tokenizer is not None:
            example["input_ids"] = self.tokenize_caption(caption).squeeze(0)

        return example
