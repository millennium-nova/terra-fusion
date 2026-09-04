# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from dataclasses import dataclass


@dataclass
class TerraFusionControlNetPipelineOutput(BaseOutput):
    textures: np.ndarray
    heightmaps: np.ndarray
    viz_images: Optional[np.ndarray] = None


class TerraFusionControlNetPipeline(StableDiffusionControlNetPipeline):
    """StableDiffusionControlNetPipeline with dual VAE decoding (texture + heightmap)."""

    def __init__(
        self,
        texture_vae: AutoencoderKL,
        heightmap_vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker=None,
        feature_extractor=None,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            texture_vae=texture_vae,
            heightmap_vae=heightmap_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        # Set required attributes
        self.vae_scale_factor = 2 ** (len(self.texture_vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # TerraFusion-specific parameters
        batch_size: int = 1,
        height_scale: int = None,
        make_viz: bool = True,
        viz_percentiles: Tuple[float, float] = (2, 98),
        **kwargs,
    ) -> TerraFusionControlNetPipelineOutput:
        if height_scale is None:
            raise ValueError("`height_scale` must be provided")

        # Run the base ControlNet pipeline to get latents
        base_output = super().__call__(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type="latent",
            return_dict=True,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )

        latents = base_output.images

        # Decode with dual VAEs: split 8-channel latents into texture (0:4) and heightmap (4:8)
        texture_latents = latents[:, :4, :, :]
        heightmap_latents = latents[:, 4:, :, :]

        tex_f = self.texture_vae.decode(
            texture_latents / self.texture_vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        tex_f = (tex_f / 2 + 0.5).clamp(0, 1)

        hgt_f = self.heightmap_vae.decode(
            heightmap_latents / self.heightmap_vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        hgt_f = (hgt_f / 2 + 0.5).clamp(0, 1)

        # Post-process
        tex_u8 = (torch.clamp(tex_f, 0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        hgt_i16 = (torch.clamp(hgt_f, 0, 1) * float(height_scale)).round().to(torch.int16).squeeze(1).cpu().numpy()

        # Visualization
        viz_np = None
        if make_viz:
            lo_p, hi_p = viz_percentiles
            viz_list = []
            tex_np = tex_u8  # (B,H,W,3)

            for b in range(hgt_f.shape[0]):
                h = hgt_f[b, 0].detach().float().cpu().numpy()  # 0..1
                lo = np.percentile(h, lo_p)
                hi = np.percentile(h, hi_p)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    h_norm = np.clip(h, 0.0, 1.0)
                else:
                    h_clip = np.clip(h, lo, hi)
                    h_norm = (h_clip - lo) / max(hi - lo, 1e-6)

                h_u8 = np.uint8(np.round(np.clip(h_norm, 0, 1) * 255.0))    # (H,W)
                h_u8_3 = np.repeat(h_u8[..., None], 3, axis=2)              # (H,W,3)
                combined = np.concatenate([tex_np[b], h_u8_3], axis=1)      # (H, Wtex+Whgt, 3)
                viz_list.append(combined)
            viz_np = np.stack(viz_list, axis=0)  # (B,H,Wtex+Whgt,3)

        if return_dict:
            return TerraFusionControlNetPipelineOutput(
                textures=tex_u8,
                heightmaps=hgt_i16,
                viz_images=viz_np if make_viz else None,
            )
        else:
            return (tex_u8, hgt_i16, viz_np) if make_viz else (tex_u8, hgt_i16)

