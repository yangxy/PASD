import os
import binascii
from safetensors import safe_open

import torch

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint
from myutils.convert_lora_safetensor_to_diffusers import convert_lora

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

def load_dreambooth_lora(unet, vae=None, text_encoder=None, model_path=None, blending_alpha=1.0, multiplier=0.6, model_base=None):
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
            if model_base is not None:
                with safe_open(model_base, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        base_state_dict[key] = f.get_tensor(key)
                                 
    if base_state_dict:
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, unet.config)
        
        unet_state_dict = unet.state_dict()
        for key in converted_unet_checkpoint:
            if key in unet_state_dict:
                converted_unet_checkpoint[key] = converted_unet_checkpoint[key] * blending_alpha + unet_state_dict[key] * (1.0 - blending_alpha)
            else:
                print(key)
        
        unet.load_state_dict(converted_unet_checkpoint, strict=False)

        if vae is not None:
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, vae.config)
            vae.load_state_dict(converted_vae_checkpoint)

        if text_encoder is not None:
            text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

    if is_lora:
        unet, text_encoder = convert_lora(unet, text_encoder, state_dict, multiplier=multiplier)
    
    return unet, vae, text_encoder