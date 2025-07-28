# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/
import inspect
import numpy as np
from typing import Callable, List, Optional, Union, Tuple

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from dataclasses import dataclass

class LatentDiffusionPipelineBase(DiffusionPipeline):
    def decode_latents(self, latents, vae): # You need to specify vae (texture or heightmap).
        latents = latents / 0.18215
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1) 
        return image.to(self.device)

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
              f"A list of generators with length {len(generator)} is provided, but the batch size is {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                  f"Unexpected latents shape: {latents.shape} expected shape: {shape}"
                )
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

@dataclass
class TerraFusionPipelineOutput(BaseOutput):
    textures: np.ndarray
    heightmaps: np.ndarray
    viz_images: Optional[np.ndarray] = None

class TerraFusionPipeline(LatentDiffusionPipelineBase):
    def __init__(
            self,
            texture_vae: AutoencoderKL,
            heightmap_vae: AutoencoderKL,
            scheduler,
            unet: UNet2DConditionModel,
            tokenizer,
            text_encoder,
    ):
        super().__init__()

        self.register_modules(
            heightmap_vae=heightmap_vae,
            texture_vae=texture_vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.texture_vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            batch_size: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            # output_type: Optional[str] = "pt",
            height_scale: int = 2000,
            make_viz: bool = True,
            viz_percentiles: Tuple[float, float] = (2, 98),
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            **kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], TerraFusionPipelineOutput]:
        # text encoding
        if isinstance(prompt, str):
            prompt = [prompt]
        if batch_size is None:
            batch_size = len(prompt)
        if len(prompt) != batch_size:
            raise ValueError(f"len(prompt)={len(prompt)} and batch_size={batch_size} mismatch.")

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        
        encoder_hidden_states = self.text_encoder(input_ids)[0].to(self.unet.dtype)

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be multiples of 8, but are {height} and {width}.")

        num_latents_ch = 8

        latents = self.prepare_latents(batch_size, num_latents_ch, height, width, self.unet.dtype, self.device, generator, latents)

        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for t in self.progress_bar(self.scheduler.timesteps):
            latents = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latents, t, encoder_hidden_states).sample

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents_texture = latents[:, :4, :, :]
        latents_height = latents[:, 4:, :, :]

        tex_f= self.decode_latents(latents_texture, self.texture_vae) # 3ch
        hgt_f = self.decode_latents(latents_height, self.heightmap_vae) # 1ch

        # ==== Post process ====
        tex_u8 = (torch.clamp(tex_f, 0, 1) * 255.0).round().to(torch.uint8).permute(0,2,3,1).cpu().numpy()
        hgt_i16 = (torch.clamp(hgt_f, 0, 1) * float(height_scale)).round().to(torch.int16).squeeze(1).cpu().numpy()

        # Visualization
        viz_np = None
        if make_viz:
            lo_p, hi_p = viz_percentiles
            viz_list = []
            tex_np = tex_u8  # (B,H,W,3)
            # calculate percentiles and normalize heightmap
            for b in range(hgt_f.shape[0]):
                h = hgt_f[b, 0].detach().float().cpu().numpy()  # 0..1
                lo = np.percentile(h, lo_p)
                hi = np.percentile(h, hi_p)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    h_norm = np.clip(h, 0.0, 1.0)
                else:
                    h_clip = np.clip(h, lo, hi)
                    h_norm = (h_clip - lo) / max(hi - lo, 1e-6)

                h_u8  = np.uint8(np.round(np.clip(h_norm, 0, 1) * 255.0))    # (H,W)
                h_u8_3 = np.repeat(h_u8[..., None], 3, axis=2)               # (H,W,3)
                combined = np.concatenate([tex_np[b], h_u8_3], axis=1)       # (H, Wtex+Whgt, 3)
                viz_list.append(combined)
            viz_np = np.stack(viz_list, axis=0)  # (B,H,Wtex+Whgt,3)

        if return_dict:
            return TerraFusionPipelineOutput(
                textures=tex_u8,
                heightmaps=hgt_i16,
                viz_images=viz_np if make_viz else None
            )
        else:
            return (tex_u8, hgt_i16, viz_np) if make_viz else (tex_u8, hgt_i16)
