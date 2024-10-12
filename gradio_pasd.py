import os
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

from pasd.pipelines.pipeline_pasd import StableDiffusionControlNetPipeline
from pasd.myutils.misc import load_dreambooth_lora, rand_name
from pasd.myutils.wavelet_color_fix import wavelet_color_fix
from pasd.annotator.retinaface import RetinaFaceDetection

use_pasd_light = False
face_detector = RetinaFaceDetection()

if use_pasd_light:
    from pasd.models.pasd_light.unet_2d_condition import UNet2DConditionModel
    from pasd.models.pasd_light.controlnet import ControlNetModel
else:
    from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel
    from pasd.models.pasd.controlnet import ControlNetModel

pretrained_model_path = "checkpoints/stable-diffusion-v1-5"
ckpt_path = "runs/pasd/checkpoint-100000"
#dreambooth_lora_path = "checkpoints/personalized_models/toonyou_beta3.safetensors"
dreambooth_lora_path = "checkpoints/personalized_models/majicmixRealistic_v6.safetensors"
#dreambooth_lora_path = "checkpoints/personalized_models/Realistic_Vision_V5.1.safetensors"
weight_dtype = torch.float16
device = "cuda"

scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_model_path}/feature_extractor")
unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
controlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="controlnet")
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
controlnet.requires_grad_(False)

unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, dreambooth_lora_path)

text_encoder.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)

validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
#validation_pipeline.enable_vae_tiling()
validation_pipeline._init_tiled_vae(decoder_tile_size=224)

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
resnet = resnet50(weights=weights)
resnet.eval()

def inference(input_image, prompt, a_prompt, n_prompt, denoise_steps, upscale, alpha, cfg, seed):
    process_size = 768
    resize_preproc = transforms.Compose([
        transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    with torch.no_grad():
        seed_everything(seed)
        generator = torch.Generator(device=device)

        input_image = input_image.convert('RGB')
        batch = preprocess(input_image).unsqueeze(0)
        prediction = resnet(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        if score >= 0.1:
            prompt += f"{category_name}" if prompt=='' else f", {category_name}"

        prompt = a_prompt if prompt=='' else f"{prompt}, {a_prompt}"

        ori_width, ori_height = input_image.size
        resize_flag = False

        rscale = upscale
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))
        
        if min(input_image.size) < process_size:
            input_image = resize_preproc(input_image)

        input_image = input_image.resize((input_image.size[0]//8*8, input_image.size[1]//8*8))
        width, height = input_image.size
        resize_flag = True #

        try:
            image = validation_pipeline(
                    None, prompt, input_image, num_inference_steps=denoise_steps, generator=generator, height=height, width=width, guidance_scale=cfg, 
                    negative_prompt=n_prompt, conditioning_scale=alpha, eta=0.0,
                ).images[0]
            
            if True: #alpha<1.0:
                image = wavelet_color_fix(image, input_image)
        
            if resize_flag: 
                image = image.resize((ori_width*rscale, ori_height*rscale))
        except Exception as e:
            print(e)
            image = Image.new(mode="RGB", size=(512, 512))

    return image

title = "Pixel-Aware Stable Diffusion for Real-ISR"
description = "Gradio Demo for PASD Real-ISR. To use it, simply upload your image, or click one of the examples to load them."
article = "<p style='text-align: center'><a href='https://github.com/yangxy/PASD' target='_blank'>Github Repo Pytorch</a></p>"
examples=[['samples/27d38eeb2dbbe7c9.png'],['samples/629e4da70703193b.png']]

demo = gr.Interface(
    fn=inference, 
    inputs=[gr.Image(type="pil"),
            gr.Textbox(label="Prompt", value="Asian"),
            gr.Textbox(label="Added Prompt", value='clean, high-resolution, 8k, best quality, masterpiece'),
            gr.Textbox(label="Negative Prompt",value='dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'),
            gr.Slider(label="Denoise Steps", minimum=10, maximum=50, value=20, step=1),
            gr.Slider(label="Upsample Scale", minimum=1, maximum=4, value=2, step=1),
            gr.Slider(label="Conditioning Scale", minimum=0.5, maximum=1.5, value=1.1, step=0.1),
            gr.Slider(label="Classier-free Guidance", minimum=0.1, maximum=10.0, value=7.5, step=0.1),
            gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)],
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    article=article,
    examples=examples).queue(concurrency_count=1)

demo.launch(
    server_name="0.0.0.0" if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1",
    share=True,
    root_path=f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv('GRADIO_PROXY_PATH') else ""
)