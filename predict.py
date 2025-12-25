# Prediction interface for Cog - PBR Texture Generator
# Based on replicate/cog-flux-schnell
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import List, Iterator
from PIL import Image
from diffusers import FluxPipeline
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)
from scipy.ndimage import sobel, gaussian_filter

MODEL_CACHE = "FLUX.1-schnell"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Texture-related keywords for validation
TEXTURE_KEYWORDS = [
    "texture", "material", "surface", "pattern", "seamless", "tileable",
    "pbr", "wood", "metal", "stone", "brick", "concrete", "fabric",
    "leather", "marble", "tile", "floor", "wall", "ground", "rock",
    "grass", "sand", "dirt", "rust", "paint", "plastic", "glass",
    "ceramic", "asphalt", "gravel", "carpet", "cloth", "paper",
    "bark", "moss", "ice", "snow", "water", "lava", "crystal",
    "scale", "skin", "fur", "grain", "woven", "knit", "plaster",
    "stucco", "terracotta", "granite", "slate", "cobblestone"
]


def is_texture_prompt(prompt: str) -> tuple:
    """Check if the prompt is texture-related."""
    prompt_lower = prompt.lower()
    found_keywords = [kw for kw in TEXTURE_KEYWORDS if kw in prompt_lower]
    if found_keywords:
        return True, ""
    warning = (
        "WARNING: Your prompt doesn't appear to be texture-related. "
        "For best PBR results, include terms like: texture, material, surface, "
        "seamless, tileable, or specific material names (wood, metal, stone, etc.)."
    )
    return False, warning


def make_seamless(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Apply seamless tiling blend to image edges."""
    if strength <= 0:
        return image

    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape[:2]

    blend_size = int(min(h, w) * 0.25 * strength)
    if blend_size < 2:
        return image

    result = img_array.copy()
    weights = np.linspace(0, 1, blend_size)

    for i, weight in enumerate(weights):
        left_col = i
        right_col = w - blend_size + i
        if len(img_array.shape) == 3:
            result[:, left_col] = (1 - weight) * img_array[:, right_col] + weight * img_array[:, left_col]
            result[:, right_col] = weight * img_array[:, left_col] + (1 - weight) * img_array[:, right_col]
        else:
            result[:, left_col] = (1 - weight) * img_array[:, right_col] + weight * img_array[:, left_col]
            result[:, right_col] = weight * img_array[:, left_col] + (1 - weight) * img_array[:, right_col]

    for i, weight in enumerate(weights):
        top_row = i
        bottom_row = h - blend_size + i
        if len(img_array.shape) == 3:
            result[top_row, :] = (1 - weight) * result[bottom_row, :] + weight * result[top_row, :]
            result[bottom_row, :] = weight * result[top_row, :] + (1 - weight) * result[bottom_row, :]
        else:
            result[top_row, :] = (1 - weight) * result[bottom_row, :] + weight * result[top_row, :]
            result[bottom_row, :] = weight * result[top_row, :] + (1 - weight) * result[bottom_row, :]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_normal_map(diffuse: Image.Image, strength: float = 1.0) -> Image.Image:
    """Generate normal map from diffuse texture using Sobel operator."""
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0
    gray = gaussian_filter(gray, sigma=0.5)

    dx = sobel(gray, axis=1) * strength
    dy = sobel(gray, axis=0) * strength
    dz = np.ones_like(gray)

    normals = np.stack([dx, -dy, dz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)

    normal_map = ((normals + 1) * 0.5 * 255).astype(np.uint8)
    return Image.fromarray(normal_map, mode="RGB")


def generate_roughness_map(diffuse: Image.Image) -> Image.Image:
    """Generate roughness map from diffuse texture."""
    gray = np.array(diffuse.convert("L"), dtype=np.float32)

    blurred = gaussian_filter(gray, sigma=3)
    local_var = gaussian_filter((gray - blurred) ** 2, sigma=5)
    local_var = local_var / (local_var.max() + 1e-8)

    intensity = 1.0 - (gray / 255.0)

    roughness = 0.5 * local_var + 0.3 * intensity + 0.2
    roughness = np.clip(roughness * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(roughness, mode="L")


def generate_ao_map(diffuse: Image.Image) -> Image.Image:
    """Generate ambient occlusion map from diffuse texture."""
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0

    ao_fine = gaussian_filter(gray, sigma=2)
    ao_medium = gaussian_filter(gray, sigma=8)
    ao_coarse = gaussian_filter(gray, sigma=16)

    ao = 0.4 * ao_fine + 0.35 * ao_medium + 0.25 * ao_coarse
    ao = (ao - ao.min()) / (ao.max() - ao.min() + 1e-8)

    ao = np.power(ao, 0.7)
    ao = 0.3 + 0.7 * ao

    ao_map = (ao * 255).astype(np.uint8)
    return Image.fromarray(ao_map, mode="L")


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Flux txt2img Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        vram = int(torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024))
        if vram < 40:
            print("GPU VRAM < 40Gb - Offloading model to CPU")
            self.txt2img_pipe.enable_model_cpu_offload()

        print("setup took: ", time.time() - start)

    @torch.amp.autocast('cuda')
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Text description of the texture to generate. Include terms like 'seamless', 'tileable', 'texture', or material names for best results.",
            default="seamless dark wood texture, highly detailed, 8k"
        ),
        resolution: int = Input(
            description="Output resolution for all texture maps",
            choices=[512, 1024, 2048],
            default=1024
        ),
        tiling_strength: float = Input(
            description="Strength of seamless tiling blend (0 = no tiling, 1 = maximum)",
            ge=0.0,
            le=1.0,
            default=0.5
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation. Use -1 for random.",
            default=-1
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.",
            default=False,
        ),
    ) -> Iterator[Path]:
        """Generate PBR texture maps from a text prompt."""

        # Validate prompt
        is_texture, warning = is_texture_prompt(prompt)
        if not is_texture:
            print(warning)

        # Handle seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Tiling strength: {tiling_strength}")

        # Enhance prompt for texture generation
        enhanced_prompt = f"{prompt}, seamless tileable texture, top-down view, flat lighting, no perspective, uniform surface"

        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate base diffuse texture
        print("Generating diffuse texture...")
        output = self.txt2img_pipe(
            prompt=[enhanced_prompt],
            width=resolution,
            height=resolution,
            guidance_scale=0.0,
            generator=generator,
            num_inference_steps=4,
            max_sequence_length=256,
            output_type="pil"
        )

        image = output.images[0]

        # Safety check
        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker([image])
            if has_nsfw_content[0]:
                raise Exception("NSFW content detected. Try a different prompt.")

        # Apply seamless tiling
        if tiling_strength > 0:
            print(f"Applying seamless tiling (strength: {tiling_strength})...")
            image = make_seamless(image, tiling_strength)

        # Save diffuse map
        diffuse_path = "/tmp/diffuse.png"
        image.save(diffuse_path)
        print("Diffuse map generated")
        yield Path(diffuse_path)

        # Generate and save normal map
        print("Generating normal map...")
        normal = generate_normal_map(image)
        if tiling_strength > 0:
            normal = make_seamless(normal, tiling_strength)
        normal_path = "/tmp/normal.png"
        normal.save(normal_path)
        print("Normal map generated")
        yield Path(normal_path)

        # Generate and save roughness map
        print("Generating roughness map...")
        roughness = generate_roughness_map(image)
        if tiling_strength > 0:
            roughness = make_seamless(roughness, tiling_strength)
        roughness_path = "/tmp/roughness.png"
        roughness.save(roughness_path)
        print("Roughness map generated")
        yield Path(roughness_path)

        # Generate and save AO map
        print("Generating ambient occlusion map...")
        ao = generate_ao_map(image)
        if tiling_strength > 0:
            ao = make_seamless(ao, tiling_strength)
        ao_path = "/tmp/ao.png"
        ao.save(ao_path)
        print("AO map generated")
        yield Path(ao_path)

        print(f"All PBR maps generated successfully! Seed: {seed}")
