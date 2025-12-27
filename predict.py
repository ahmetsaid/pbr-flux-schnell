# PBR Texture Generator - Playground v2.5 version
from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from scipy.ndimage import sobel, gaussian_filter


def make_seamless(image: Image.Image, strength: float = 0.5) -> Image.Image:
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
        left_col, right_col = i, w - blend_size + i
        if len(img_array.shape) == 3:
            result[:, left_col] = (1 - weight) * img_array[:, right_col] + weight * img_array[:, left_col]
            result[:, right_col] = weight * img_array[:, left_col] + (1 - weight) * img_array[:, right_col]
    for i, weight in enumerate(weights):
        top_row, bottom_row = i, h - blend_size + i
        if len(img_array.shape) == 3:
            result[top_row, :] = (1 - weight) * result[bottom_row, :] + weight * result[top_row, :]
            result[bottom_row, :] = weight * result[top_row, :] + (1 - weight) * result[bottom_row, :]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_normal_map(diffuse: Image.Image, strength: float = 1.0) -> Image.Image:
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0
    gray = gaussian_filter(gray, sigma=0.5)
    dx = sobel(gray, axis=1) * strength
    dy = sobel(gray, axis=0) * strength
    dz = np.ones_like(gray)
    normals = np.stack([dx, -dy, dz], axis=-1)
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    normal_map = ((normals + 1) * 0.5 * 255).astype(np.uint8)
    return Image.fromarray(normal_map, mode="RGB")


def generate_roughness_map(diffuse: Image.Image) -> Image.Image:
    gray = np.array(diffuse.convert("L"), dtype=np.float32)
    blurred = gaussian_filter(gray, sigma=3)
    local_var = gaussian_filter((gray - blurred) ** 2, sigma=5)
    local_var = local_var / (local_var.max() + 1e-8)
    intensity = 1.0 - (gray / 255.0)
    roughness = 0.5 * local_var + 0.3 * intensity + 0.2
    roughness = np.clip(roughness * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(roughness, mode="L")


def generate_ao_map(diffuse: Image.Image) -> Image.Image:
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0
    ao_fine = gaussian_filter(gray, sigma=2)
    ao_medium = gaussian_filter(gray, sigma=8)
    ao_coarse = gaussian_filter(gray, sigma=16)
    ao = 0.4 * ao_fine + 0.35 * ao_medium + 0.25 * ao_coarse
    ao = (ao - ao.min()) / (ao.max() - ao.min() + 1e-8)
    ao = np.power(ao, 0.7)
    ao = 0.3 + 0.7 * ao
    return Image.fromarray((ao * 255).astype(np.uint8), mode="L")


def create_grid(diffuse: Image.Image, normal: Image.Image, roughness: Image.Image, ao: Image.Image) -> Image.Image:
    """Create 2x2 grid: color, normal, roughness, ao"""
    w, h = diffuse.size
    roughness_rgb = roughness.convert("RGB")
    ao_rgb = ao.convert("RGB")
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(diffuse, (0, 0))
    grid.paste(normal, (w, 0))
    grid.paste(roughness_rgb, (0, h))
    grid.paste(ao_rgb, (w, h))
    return grid


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Loading Playground v2.5...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True
        )
        self.pipe.to("cuda")
        print("Model ready!")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Text description of the texture",
            default="seamless dark wood texture, highly detailed, 8k"
        ),
        resolution: int = Input(
            description="Output resolution",
            choices=[512, 1024],
            default=1024
        ),
        tiling_strength: float = Input(
            description="Seamless tiling strength (0-1)",
            ge=0.0, le=1.0, default=0.5
        ),
        num_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=50, default=25
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1
        ),
        output_format: str = Input(
            description="Output format",
            choices=["grid", "separate"],
            default="separate"
        ),
    ) -> list[Path]:
        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Seed: {seed}, Resolution: {resolution}, Steps: {num_steps}")

        enhanced_prompt = f"{prompt}, seamless tileable texture, top-down view, flat lighting, PBR material"
        generator = torch.Generator("cuda").manual_seed(seed)

        print("Generating color...")
        output = self.pipe(
            prompt=enhanced_prompt,
            width=resolution,
            height=resolution,
            num_inference_steps=num_steps,
            guidance_scale=3.0,
            generator=generator,
        )
        image = output.images[0]

        if tiling_strength > 0:
            image = make_seamless(image, tiling_strength)

        print("Generating normal...")
        normal = generate_normal_map(image)
        if tiling_strength > 0:
            normal = make_seamless(normal, tiling_strength)

        print("Generating roughness...")
        roughness = generate_roughness_map(image)
        if tiling_strength > 0:
            roughness = make_seamless(roughness, tiling_strength)

        print("Generating AO...")
        ao = generate_ao_map(image)
        if tiling_strength > 0:
            ao = make_seamless(ao, tiling_strength)

        print(f"Done! Seed: {seed}")

        if output_format == "grid":
            grid = create_grid(image, normal, roughness, ao)
            grid_path = "/tmp/pbr_grid.png"
            grid.save(grid_path)
            return [Path(grid_path)]
        else:
            color_path = "/tmp/color.png"
            normal_path = "/tmp/normal.png"
            roughness_path = "/tmp/roughness.png"
            ao_path = "/tmp/ao.png"

            image.save(color_path)
            normal.save(normal_path)
            roughness.save(roughness_path)
            ao.save(ao_path)

            return [Path(color_path), Path(normal_path), Path(roughness_path), Path(ao_path)]
