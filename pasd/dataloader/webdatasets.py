import cv2
import math
import random
import itertools
import torch
from torch import nn
from typing import List, Union
from braceexpand import braceexpand
from functools import partial
import webdataset as wds
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import default_collate
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from .realesrgan import RealESRGAN_degradation
from ..myutils.img_util import convert_image_to_fn
from ..myutils.misc import exists

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 512,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        control_type: str = "realisr",
        tokenizer = None,
        null_text_ratio: float = 0.1,
        convert_image_to: str = "RGB",
        center_crop: bool = False,
        random_flip: bool = True,
        resize_bak: bool = True,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        crop_preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        img_preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        def transform(example):
            if crop_preproc is not None:
                image = crop_preproc(example["image"])
                
                example["image"] = img_preproc(image)
                if control_type is not None:
                    if control_type == 'realisr':
                        GT_image_t, LR_image_t = degradation.degrade_process(np.asarray(image)/255., resize_bak=resize_bak)
                        example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
                        example["image"] = GT_image_t.squeeze(0) * 2.0 - 1.0
                    elif control_type == 'grayscale':
                        image = np.asarray(image.convert('L').convert('RGB'))/255.
                        example["conditioning_pixel_values"] = torch.from_numpy(image).permute(2,0,1)
                    else:
                        raise NotImplementedError

            caption = example['text'] if 'text' in example else ''
            if tokenizer is not None:
                example["input_ids"] = tokenize_caption(caption, tokenizer, null_text_ratio).squeeze(0)

            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys({"image", "text"})),
            wds.map(transform),
            wds.to_tuple("image", "text", "input_ids", "conditioning_pixel_values"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

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

def tokenize_caption(caption, tokenizer, null_text_ratio):
    if random.random() < null_text_ratio:
        caption = ""
                    
    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    return inputs.input_ids

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
        keys = ["image", 'text'] + extra_keys
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
        self.append(wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=handler)),
        self.append(key_verifier(required_keys=keys, handler=handler))

        # Apply preprocessing
        self.append(wds.map(self.preproc))
        self.append(wds.to_tuple("image", "text", "input_ids", "conditioning_pixel_values")),

    def preproc(self, example):
        """Applies the preprocessing for images"""
        if self.crop_preproc is not None:
            image = self.crop_preproc(example["image"])
            
            example["image"] = self.img_preproc(image)
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

        caption = example['text'] if 'text' in example else ''
        if self.tokenizer is not None:
            example["input_ids"] = tokenize_caption(caption, self.tokenizer, self.null_text_ratio).squeeze(0)

        return example
