import os
import binascii
from safetensors import safe_open

import torch

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def identity(x):
    return x

def load_dreambooth_lora(unet, vae=None, model_path=None, alpha=1.0, model_base=""):
    if model_path is None: return unet
    
    if model_path.endswith(".ckpt"):
        base_state_dict = torch.load(model_path)['state_dict']
    elif model_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
                            
        is_lora = all("lora" in k for k in state_dict.keys())
        if not is_lora:
            base_state_dict = state_dict
        else:
            base_state_dict = {}
            with safe_open(model_base, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_state_dict[key] = f.get_tensor(key)
                                 
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, unet.config)
    unet_state_dict = unet.state_dict()
    for key in converted_unet_checkpoint:
        converted_unet_checkpoint[key] = alpha * converted_unet_checkpoint[key] + (1.0-alpha) * unet_state_dict[key]
    unet.load_state_dict(converted_unet_checkpoint, strict=False)

    if vae is not None:
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, vae.config)
        vae.load_state_dict(converted_vae_checkpoint)
    
    return unet, vae