# PBR Texture Generator v2 - AI Depth Estimation
from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from typing import Literal

from pbr_maps import (
    DepthEstimator,
    HeightProcessor,
    NormalGenerator,
    AOGenerator,
    RoughnessGenerator,
    EmissiveGenerator,
    SeamlessTiling,
    save_height_16bit
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load models once during container startup"""
        print("Loading Playground v2.5...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True
        )
        self.pipe.to("cuda")

        print("Loading Intel DPT-Large for depth estimation...")
        self.depth_estimator = DepthEstimator(
            model_name="Intel/dpt-large",
            device="cuda",
            local_files_only=True
        )

        # Initialize processing classes (stateless)
        self.height_processor = HeightProcessor()
        self.normal_generator = NormalGenerator()
        self.ao_generator = AOGenerator()
        self.roughness_generator = RoughnessGenerator()
        self.emissive_generator = EmissiveGenerator()
        self.seamless_tiler = SeamlessTiling()

        print("All models ready!")

    @torch.inference_mode()
    def predict(
        self,
        # Input options
        input_image: Path = Input(
            description="Optional: Upload your own texture image to generate PBR maps (skips AI generation)",
            default=None
        ),
        prompt: str = Input(
            description="Text description of the texture (ignored if input_image is provided)",
            default="seamless dark wood texture, highly detailed, 8k"
        ),
        negative_prompt: str = Input(
            description="Things to avoid in the texture",
            default=""
        ),
        resolution: int = Input(
            description="Output resolution (only used when generating from prompt)",
            choices=[512, 1024],
            default=1024
        ),
        num_steps: int = Input(
            description="Number of inference steps (only used when generating from prompt)",
            ge=1, le=50, default=25
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1
        ),

        # Tiling controls
        tiling_strength: float = Input(
            description="Seamless tiling strength (0-1)",
            ge=0.0, le=1.0, default=0.5
        ),

        # Height map controls
        height_contrast: float = Input(
            description="Height map contrast adjustment",
            ge=0.5, le=3.0, default=1.0
        ),
        height_gamma: float = Input(
            description="Height map gamma (>1 darkens, <1 lightens)",
            ge=0.3, le=3.0, default=1.0
        ),
        suppress_scene_depth: bool = Input(
            description="Suppress large-scale depth variations (recommended for textures)",
            default=True
        ),

        # Normal map controls
        normal_strength: float = Input(
            description="Normal map intensity",
            ge=0.1, le=5.0, default=1.0
        ),
        normal_detail: float = Input(
            description="Blend in fine detail from color (0=height only, 1=max detail)",
            ge=0.0, le=1.0, default=0.25
        ),
        normal_format: str = Input(
            description="Normal map format (opengl=Y+ up for Blender/Unreal, directx=Y- for Unity)",
            choices=["opengl", "directx"],
            default="opengl"
        ),

        # AO controls
        ao_strength: float = Input(
            description="Ambient occlusion intensity",
            ge=0.0, le=2.0, default=1.0
        ),
        ao_radius: float = Input(
            description="AO sampling radius (affects shadow spread)",
            ge=1.0, le=32.0, default=8.0
        ),

        # Roughness controls
        roughness_contrast: float = Input(
            description="Roughness map contrast",
            ge=0.5, le=2.0, default=1.0
        ),
        roughness_base: float = Input(
            description="Base roughness level (0=smooth, 1=rough)",
            ge=0.0, le=1.0, default=0.5
        ),

        # Output options
        output_16bit_height: bool = Input(
            description="Export height map as 16-bit PNG (higher precision)",
            default=False
        ),
    ) -> list[Path]:
        """Generate PBR texture maps with AI depth estimation"""

        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")

        # Step 1: Get or generate base image
        if input_image is not None:
            print("Using uploaded image...")
            image = Image.open(str(input_image)).convert("RGB")
            print(f"Input image size: {image.size}")
        else:
            print(f"Generating texture (seed={seed}, resolution={resolution}, steps={num_steps})...")
            image = self._generate_texture(prompt, negative_prompt, resolution, num_steps, seed)

        # Step 2: Estimate depth using DPT
        print("Estimating depth with AI...")
        raw_depth = self.depth_estimator.estimate_depth(image)

        # Step 3: Convert depth to height map
        print("Processing height map...")
        height = self.height_processor.depth_to_height(
            raw_depth,
            suppress_scene_depth=suppress_scene_depth
        )
        height = self.height_processor.apply_contrast_gamma(
            height,
            contrast=height_contrast,
            gamma=height_gamma
        )

        # Step 4: Make height map tile-safe BEFORE deriving other maps
        if tiling_strength > 0:
            print("Making maps seamless...")
            height = self.height_processor.make_tile_safe(height, strength=tiling_strength)
            image = self.seamless_tiler.make_seamless(image, strength=tiling_strength)

        # Step 5: Generate derived maps from tile-safe height
        print("Generating normal map...")
        normal = self.normal_generator.height_to_normal(
            height,
            strength=normal_strength,
            format=normal_format,
            color_image=image,
            detail_blend=normal_detail
        )

        print("Generating AO map...")
        ao = self.ao_generator.height_to_ao(
            height,
            strength=ao_strength,
            radius=ao_radius
        )

        print("Generating roughness map...")
        roughness = self.roughness_generator.estimate_roughness(
            image,
            height,
            contrast=roughness_contrast,
            base_roughness=roughness_base
        )

        print("Generating emissive map...")
        emissive = self.emissive_generator.estimate_emissive(image)

        # Step 6: Save outputs
        output_paths = self._save_outputs(
            image, height, normal, ao, roughness, emissive,
            output_16bit_height=output_16bit_height
        )

        print(f"Done! Seed: {seed}")
        return output_paths

    def _generate_texture(self, prompt, negative_prompt, resolution, num_steps, seed):
        """Generate texture using Playground v2.5"""
        enhanced_prompt = f"{prompt}, seamless tileable texture, top-down view, flat lighting, PBR material"
        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=resolution,
            height=resolution,
            num_inference_steps=num_steps,
            guidance_scale=3.0,
            generator=generator,
        )
        return output.images[0]

    def _save_outputs(self, image, height, normal, ao, roughness, emissive, output_16bit_height):
        """Save all output files"""
        paths = []

        # Color
        color_path = "/tmp/color.png"
        image.save(color_path)
        paths.append(Path(color_path))

        # Height (8-bit or 16-bit)
        height_path = "/tmp/height.png"
        if output_16bit_height:
            save_height_16bit(height, height_path)
        else:
            height_img = Image.fromarray((height * 255).astype(np.uint8), mode="L")
            height_img.save(height_path)
        paths.append(Path(height_path))

        # Normal
        normal_path = "/tmp/normal.png"
        normal.save(normal_path)
        paths.append(Path(normal_path))

        # Roughness
        roughness_path = "/tmp/roughness.png"
        roughness.save(roughness_path)
        paths.append(Path(roughness_path))

        # AO
        ao_path = "/tmp/ao.png"
        ao.save(ao_path)
        paths.append(Path(ao_path))

        # Emissive
        emissive_path = "/tmp/emissive.png"
        emissive.save(emissive_path)
        paths.append(Path(emissive_path))

        # Grid preview (3x2)
        grid_path = "/tmp/grid.png"
        grid = self._create_grid(image, height, normal, roughness, ao, emissive)
        grid.save(grid_path)
        paths.append(Path(grid_path))

        return paths

    def _create_grid(self, image, height, normal, roughness, ao, emissive):
        """Create 3x2 preview grid: color, height, normal (top) | roughness, ao, emissive (bottom)"""
        w, h = image.size
        height_img = Image.fromarray((height * 255).astype(np.uint8), mode="L").convert("RGB")
        roughness_rgb = roughness.convert("RGB")
        ao_rgb = ao.convert("RGB")
        emissive_rgb = emissive.convert("RGB")

        # 3x2 grid
        grid = Image.new("RGB", (w * 3, h * 2), color=(40, 40, 40))
        grid.paste(image, (0, 0))
        grid.paste(height_img, (w, 0))
        grid.paste(normal, (w * 2, 0))
        grid.paste(roughness_rgb, (0, h))
        grid.paste(ao_rgb, (w, h))
        grid.paste(emissive_rgb, (w * 2, h))

        return grid
