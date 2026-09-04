# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from src.pipeline_controlnet import TerraFusionControlNetPipeline
from src.renderer import render_terrain_3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
import argparse
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ===== Parse arguments =====
parser = argparse.ArgumentParser(description="Sketch-based terrain generation using TerraFusion ControlNet.")
parser.add_argument("--sketch_image", type=str, required=True, help="Path to sketch image (PNG).")
parser.add_argument("--output_dir", type=str, default=f"./inference_outputs/{current_time}", help="Path to output directory.")
parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate from the same sketch.")
parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
parser.add_argument("--num_inference_steps", type=int, default=200, help="Number of denoising steps.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Omit for a random seed.")
parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale.")
args = parser.parse_args()

output_dir = args.output_dir
combined_image_dir = os.path.join(output_dir, "combined")  # for visualization
texture_dir = os.path.join(output_dir, "texture")  # texture (uint8, png)
heightmap_dir = os.path.join(output_dir, "heightmap")  # heightmap (int16, tif)
render_dir = os.path.join(output_dir, "render")  # 3D render (uint8, png)
num_samples = args.num_samples
batch_size = args.batch_size

prompt = "A satellite terrain image."  # text prompt (fixed)
# Note: Our current model does not allow free prompts.

# ===== Load sketch image =====
print(f"Loading sketch image: {args.sketch_image}")
sketch_image = Image.open(args.sketch_image).convert("RGB")
resolution = 512
preprocess = transforms.Compose([
    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
])
input_sketch = preprocess(sketch_image)  # (3, 512, 512)
print(f"Sketch preprocessed: {input_sketch.shape}")

# ===== Load pretrained models =====
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

# Tokenizer and TextEncoder
tokenizer = CLIPTokenizer.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="text_encoder").to(device, dtype=weight_dtype)

# VAE
texture_vae = AutoencoderKL.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="texture_vae").to(device, dtype=weight_dtype)
heightmap_vae = AutoencoderKL.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="heightmap_vae").to(device, dtype=weight_dtype)

# UNet
unet = UNet2DConditionModel.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="unet").to(device, dtype=weight_dtype)

# Noise scheduler
scheduler = DDPMScheduler.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="scheduler")

# ControlNet
controlnet = ControlNetModel.from_pretrained("Millennium-Nova/terra-fusion-controlnet").to(device, dtype=weight_dtype)

# ===== Initialize pipeline =====
pipeline = TerraFusionControlNetPipeline(
    texture_vae=texture_vae,
    heightmap_vae=heightmap_vae,
    scheduler=scheduler,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    controlnet=controlnet,
)
pipeline.to(device)
print("Models loaded successfully.")

# ===== Setup generator =====
generator = torch.Generator(device=pipeline.device)
if args.seed is not None:
    generator.manual_seed(args.seed)
    print(f"Using seed: {args.seed}")
else:
    generator.seed()
    print(f"Using random seed: {generator.initial_seed()}")

# ===== Create output directories =====
os.makedirs(combined_image_dir, exist_ok=True)
os.makedirs(texture_dir, exist_ok=True)
os.makedirs(heightmap_dir, exist_ok=True)
os.makedirs(render_dir, exist_ok=True)

# ===== Generate sample images =====
num_batches = (num_samples + batch_size - 1) // batch_size

with tqdm(total=num_samples, desc="Generating samples", ncols=100) as pbar:
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        prompts = [prompt] * current_batch_size

        # Repeat sketch for batch
        batch_sketch = input_sketch.unsqueeze(0).repeat(current_batch_size, 1, 1, 1).to(device)

        with torch.no_grad():
            outputs = pipeline(
                prompt=prompts,
                image=batch_sketch,
                batch_size=current_batch_size,
                num_inference_steps=args.num_inference_steps,
                make_viz=True,
                height_scale=2000,
                generator=generator,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
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

            render_file_path = os.path.join(render_dir, f"render_{idx_str}.png")
            render_terrain_3d(outputs.textures[i], outputs.heightmaps[i], render_file_path)

            # Visualization: (H, W_tex+W_hgt, 3) uint8 → PNG
            if outputs.viz_images is not None:
                combined_image_file_path = os.path.join(combined_image_dir, f"combined_{idx_str}.png")
                imageio.imwrite(combined_image_file_path, outputs.viz_images[i])
                print(f"Saved: texture_{idx_str}.png, heightmap_{idx_str}.tif, combined_{idx_str}.png, render_{idx_str}.png")
            else:
                print(f"Saved: texture_{idx_str}.png, heightmap_{idx_str}.tif, render_{idx_str}.png")

            pbar.update(1)

print(f"Generated {num_samples} samples from sketch: {args.sketch_image}")

