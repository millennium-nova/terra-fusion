# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from src.pipeline import TerraFusionPipeline 
from tqdm import tqdm 
import imageio.v2 as imageio
import numpy as np
import argparse
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ===== Parse arguments =====
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=f"./inference_outputs/{current_time}", help="Path to output directory.")
parser.add_argument("--model_dir", type=str, default="./models", help="Path to model directory.")
parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate.")
parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")
args = parser.parse_args()

output_dir = args.output_dir
combined_image_dir = os.path.join(output_dir, "combined") # for visualization
texture_dir = os.path.join(output_dir, "texture") # texture (uint8, png)
heightmap_dir = os.path.join(output_dir, "heightmap") # heightmap (int16, tif)
model_dir = args.model_dir
num_samples = args.num_samples
batch_size = args.batch_size


prompt = "A satellite terrain image." # text prompt
# Note: Our current model does not allow free prompts.

# ===== Load pretrained models =====
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer and TextEncoder
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_dir, "text_encoder")).to(device)

# VAE
texture_vae = AutoencoderKL.from_pretrained(os.path.join(model_dir, "texture_vae")).to(device)
heightmap_vae = AutoencoderKL.from_pretrained(os.path.join(model_dir, "heightmap_vae")).to(device)

# UNet
unet = UNet2DConditionModel.from_pretrained(os.path.join(model_dir, "unet")).to(device)

# Noise scheduler
scheduler = DDPMScheduler.from_pretrained(os.path.join(model_dir, "scheduler"))

# ===== Initialize pipeline =====
pipeline = TerraFusionPipeline(
    texture_vae=texture_vae,
    heightmap_vae=heightmap_vae,
    scheduler=scheduler,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
)
pipeline.to(device)

generator = torch.Generator(device=pipeline.device)

# ===== Generate sample images =====
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(combined_image_dir):
    os.makedirs(combined_image_dir)
if not os.path.exists(texture_dir):
    os.makedirs(texture_dir)
if not os.path.exists(heightmap_dir):
    os.makedirs(heightmap_dir)


num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches

with tqdm(total=num_samples, desc="Generating samples", ncols=100) as pbar: 
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        prompts = [prompt] * current_batch_size 

        with torch.no_grad():
            outputs = pipeline(
                prompt=prompts,
                batch_size=current_batch_size,
                num_inference_steps=args.num_inference_steps,
                make_viz=True,
                height_scale=2000,
                generator=generator,
            ) 

        pad = max(6, len(str(num_samples)))
        # Save images
        for i in range(current_batch_size):
            idx = batch_idx * batch_size + i
            idx_str = f"{idx:0{pad}d}"

            # Texture: (H, W, 3) uint8 → PNG
            texture_file_path = os.path.join(texture_dir, f"texture_{idx_str}.png")
            imageio.imwrite(texture_file_path, outputs.textures[i])

            # Heightmap: (H, W) int16 → TIF
            heightmap_file_path = os.path.join(heightmap_dir, f"heightmap_{idx_str}.tif")
            imageio.imwrite(heightmap_file_path, outputs.heightmaps[i])

            # Visualization: (H, W_tex+W_hgt, 3) uint8 → PNG（when make_viz=True）
            if outputs.viz_images is not None:
                combined_image_file_path = os.path.join(combined_image_dir, f"combined_{idx_str}.png")
                imageio.imwrite(combined_image_file_path, outputs.viz_images[i])
                print(f"Saved: {texture_file_path}, {heightmap_file_path}, and {combined_image_file_path}")
            else:
                print(f"Saved: {texture_file_path}, {heightmap_file_path}")

            pbar.update(1)

print(f"Generated {num_samples} samples.")
