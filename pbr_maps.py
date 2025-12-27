# pbr_maps.py - PBR Map Generation Module
"""
AI-based PBR texture map generation using Intel DPT-Large.
All derived maps (normal, AO, roughness) are computed deterministically
from the AI-estimated height map.
"""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from typing import Literal
import cv2


# =============================================================================
# DEPTH ESTIMATION
# =============================================================================

class DepthEstimator:
    """
    Wraps Intel DPT-Large model for monocular depth estimation.

    Reference: https://huggingface.co/Intel/dpt-large
    """

    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        device: str = "cuda",
        local_files_only: bool = True
    ):
        from transformers import DPTImageProcessor, DPTForDepthEstimation

        self.device = device
        self.processor = DPTImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )
        self.model = DPTForDepthEstimation.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth from RGB image.

        Args:
            image: PIL RGB image

        Returns:
            np.ndarray: Depth map as float32, shape (H, W),
                       higher values = farther from camera
        """
        original_size = image.size  # (W, H)

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H', W')

        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False
        )

        return depth.squeeze().cpu().numpy().astype(np.float32)


# =============================================================================
# HEIGHT MAP PROCESSING
# =============================================================================

class HeightProcessor:
    """
    Processes raw depth maps into usable height maps for PBR workflows.
    """

    def depth_to_height(
        self,
        depth: np.ndarray,
        suppress_scene_depth: bool = True,
        high_pass_sigma: float = 50.0
    ) -> np.ndarray:
        """
        Convert depth map to height map.

        DPT outputs "depth" (distance from camera), but we want "height"
        (displacement from surface). For flat-on textures, we invert and
        optionally high-pass filter to remove large-scale scene assumptions.

        Args:
            depth: Raw depth from DPT, float32
            suppress_scene_depth: If True, applies high-pass filter to remove
                                 global depth trends (recommended for textures)
            high_pass_sigma: Sigma for high-pass Gaussian filter

        Returns:
            Height map normalized to [0, 1], float32
        """
        # Invert: farther = lower height for typical top-down textures
        height = -depth

        if suppress_scene_depth:
            # High-pass filter to preserve detail, remove scene-scale depth
            # This removes the "scene depth" that DPT tends to add
            low_freq = gaussian_filter(height, sigma=high_pass_sigma)
            height = height - low_freq

        # Normalize to [0, 1]
        h_min, h_max = height.min(), height.max()
        if h_max - h_min > 1e-8:
            height = (height - h_min) / (h_max - h_min)
        else:
            height = np.full_like(height, 0.5)

        return height.astype(np.float32)

    def apply_contrast_gamma(
        self,
        height: np.ndarray,
        contrast: float = 1.0,
        gamma: float = 1.0
    ) -> np.ndarray:
        """
        Apply user-controlled contrast and gamma to height map.

        Args:
            height: Normalized height map [0, 1]
            contrast: Multiplier centered at 0.5 (1.0 = no change)
            gamma: Gamma curve (>1 darkens, <1 lightens)

        Returns:
            Adjusted height map, clipped to [0, 1]
        """
        # Contrast adjustment centered at 0.5
        height = (height - 0.5) * contrast + 0.5

        # Gamma adjustment
        height = np.clip(height, 0, 1)
        height = np.power(height, gamma)

        return height.astype(np.float32)

    def make_tile_safe(
        self,
        height: np.ndarray,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Make height map seamlessly tileable using offset method.

        Uses the classic "offset and blend" technique:
        1. Shift image by 50% to put edges in center
        2. Blend the center seams with smooth falloff
        3. Result tiles perfectly at original edges

        Args:
            height: Height map [0, 1]
            strength: Blend intensity (0-1)

        Returns:
            Tile-safe height map
        """
        if strength <= 0:
            return height

        h, w = height.shape
        blend_width = int(min(h, w) * 0.15 * strength)  # Blend region
        if blend_width < 4:
            return height

        # Step 1: Shift image by 50% - seams now in center
        shifted = np.roll(np.roll(height, h // 2, axis=0), w // 2, axis=1)

        # Step 2: Create smooth blend mask for center cross
        mask = self._create_center_blend_mask(h, w, blend_width)

        # Step 3: Blend shifted with original at center
        # Where mask=1, use original (edges); where mask=0, use shifted (center)
        result = shifted * (1 - mask) + height * mask

        # Step 4: Shift back
        result = np.roll(np.roll(result, -h // 2, axis=0), -w // 2, axis=1)

        return result.astype(np.float32)

    def _create_center_blend_mask(self, h: int, w: int, blend_width: int) -> np.ndarray:
        """
        Create a mask that's 0 at center cross, 1 at edges.
        Uses smoothstep for artifact-free blending.
        """
        # Horizontal distance from center
        x = np.abs(np.arange(w) - w // 2)
        x_mask = self._smoothstep(x, blend_width // 2, blend_width)

        # Vertical distance from center
        y = np.abs(np.arange(h) - h // 2)
        y_mask = self._smoothstep(y, blend_width // 2, blend_width)

        # Combine: both must be away from center
        mask = np.outer(y_mask, np.ones(w)) * np.outer(np.ones(h), x_mask)

        return mask

    def _smoothstep(self, x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
        """Smooth Hermite interpolation between 0 and 1."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0, 1)
        return t * t * (3 - 2 * t)


# =============================================================================
# NORMAL MAP GENERATION
# =============================================================================

class NormalGenerator:
    """
    Generate normal maps from height maps using Sobel operators.
    Optionally blend with color-derived normals for fine detail.

    Reference:
    - https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    """

    def height_to_normal(
        self,
        height: np.ndarray,
        strength: float = 1.0,
        format: Literal["opengl", "directx"] = "opengl",
        color_image: Image.Image = None,
        detail_blend: float = 0.25
    ) -> Image.Image:
        """
        Convert height map to normal map using Sobel derivatives.
        Optionally blends in high-frequency detail from color image.

        Args:
            height: Height map [0, 1], float32
            strength: Normal intensity multiplier
            format: "opengl" (Y+ up) or "directx" (Y- up)
            color_image: Optional RGB image for fine detail extraction
            detail_blend: How much color-based detail to blend (0-1)

        Returns:
            PIL RGB image with normal map
        """
        # Generate normals from height (smooth, large-scale)
        height_normals = self._compute_normals(height, strength, format, sigma=0.5)

        # If color image provided, blend in fine detail
        if color_image is not None and detail_blend > 0:
            gray = np.array(color_image.convert("L"), dtype=np.float32) / 255.0
            # Use smaller sigma for sharper detail from color
            color_normals = self._compute_normals(gray, strength * 1.5, format, sigma=0.3)

            # Blend: height for large-scale, color for fine detail
            normals = height_normals * (1 - detail_blend) + color_normals * detail_blend

            # Re-normalize after blending
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / (norms + 1e-8)
        else:
            normals = height_normals

        # Convert from [-1, 1] to [0, 255]
        normal_rgb = ((normals + 1) * 0.5 * 255).astype(np.uint8)

        return Image.fromarray(normal_rgb, mode="RGB")

    def _compute_normals(
        self,
        source: np.ndarray,
        strength: float,
        format: str,
        sigma: float = 0.5
    ) -> np.ndarray:
        """
        Compute normal vectors from a grayscale source using Sobel.

        Returns:
            Normalized vectors as (H, W, 3) array in range [-1, 1]
        """
        # Apply blur to control detail level
        smoothed = gaussian_filter(source, sigma=sigma)

        # Compute partial derivatives using Sobel
        dx = sobel(smoothed, axis=1) * strength
        dy = sobel(smoothed, axis=0) * strength

        # For OpenGL: +Y is up, so we negate dy
        if format == "opengl":
            dy = -dy

        # Z component
        dz = np.ones_like(source)

        # Stack into (H, W, 3)
        normals = np.stack([dx, dy, dz], axis=-1)

        # Normalize each vector to unit length
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / (norms + 1e-8)

        return normals


# =============================================================================
# AMBIENT OCCLUSION GENERATION
# =============================================================================

class AOGenerator:
    """
    Generate ambient occlusion from height maps.

    Uses a multi-scale approach that simulates how recessed areas
    receive less ambient light.
    """

    def height_to_ao(
        self,
        height: np.ndarray,
        strength: float = 1.0,
        radius: float = 8.0,
        num_scales: int = 4
    ) -> Image.Image:
        """
        Generate AO map from height map.

        Uses multi-scale analysis: areas that are lower than their
        surroundings at multiple scales are considered occluded.

        Args:
            height: Height map [0, 1]
            strength: AO intensity multiplier
            radius: Base sampling radius (affects shadow spread)
            num_scales: Number of blur scales to combine

        Returns:
            PIL grayscale image (white = full light, black = occluded)
        """
        ao_combined = np.zeros_like(height)

        # Multi-scale occlusion
        for i in range(num_scales):
            scale_radius = radius * (2 ** i)

            # Local average height at this scale
            local_avg = gaussian_filter(height, sigma=scale_radius)

            # Occlusion = how much lower than average
            occlusion = local_avg - height

            # Only count actual occlusion (where point is lower)
            occlusion = np.maximum(occlusion, 0)

            # Weight by scale (closer scales matter more)
            weight = 1.0 / (i + 1)
            ao_combined += occlusion * weight

        # Normalize and invert
        max_ao = ao_combined.max()
        if max_ao > 1e-8:
            ao_combined = ao_combined / max_ao
        ao = 1.0 - (ao_combined * strength)

        # Clamp and add minimum light level
        ao = np.clip(ao, 0.1, 1.0)

        # Apply subtle gamma for natural shadows
        ao = np.power(ao, 0.8)

        return Image.fromarray((ao * 255).astype(np.uint8), mode="L")


# =============================================================================
# ROUGHNESS GENERATION
# =============================================================================

class RoughnessGenerator:
    """
    Generate roughness maps from RGB image and height map.

    Combines multiple signals:
    1. Local texture variance (high detail = rougher)
    2. Height variation (bumpy areas tend to be rougher)
    3. Color intensity (darker areas often rougher in natural materials)
    """

    def estimate_roughness(
        self,
        image: Image.Image,
        height: np.ndarray,
        contrast: float = 1.0,
        base_roughness: float = 0.5,
        detail_weight: float = 0.4,
        height_weight: float = 0.3,
        intensity_weight: float = 0.2
    ) -> Image.Image:
        """
        Estimate roughness from RGB + height.

        Args:
            image: RGB PIL image
            height: Height map [0, 1]
            contrast: Roughness contrast multiplier
            base_roughness: Base roughness level [0, 1]
            detail_weight: Weight for texture detail
            height_weight: Weight for height variation
            intensity_weight: Weight for intensity

        Returns:
            PIL grayscale roughness map (white = rough, black = smooth)
        """
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0

        # Component 1: Local texture variance (detail)
        blurred = gaussian_filter(gray, sigma=3)
        local_var = gaussian_filter((gray - blurred) ** 2, sigma=5)
        max_var = local_var.max()
        if max_var > 1e-8:
            local_var = local_var / max_var

        # Component 2: Height variation (gradient magnitude)
        dx = sobel(height, axis=1)
        dy = sobel(height, axis=0)
        height_grad = np.sqrt(dx**2 + dy**2)
        max_grad = height_grad.max()
        if max_grad > 1e-8:
            height_grad = height_grad / max_grad

        # Component 3: Intensity (darker = rougher heuristic)
        intensity = 1.0 - gray

        # Combine with weights
        remaining_weight = 1.0 - detail_weight - height_weight - intensity_weight
        roughness = (
            detail_weight * local_var +
            height_weight * height_grad +
            intensity_weight * intensity +
            remaining_weight * base_roughness
        )

        # Apply contrast centered at base_roughness
        roughness = (roughness - base_roughness) * contrast + base_roughness
        roughness = np.clip(roughness, 0, 1)

        return Image.fromarray((roughness * 255).astype(np.uint8), mode="L")


# =============================================================================
# SEAMLESS TILING UTILITIES
# =============================================================================

class SeamlessTiling:
    """
    Make RGB images seamlessly tileable using offset method.
    """

    def make_seamless(
        self,
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """
        Make RGB image seamlessly tileable using offset method.

        Uses the classic "offset and blend" technique:
        1. Shift image by 50% to put edges in center
        2. Blend the center seams with smooth falloff
        3. Result tiles perfectly at original edges

        Args:
            image: RGB PIL image
            strength: Blend strength (0-1)

        Returns:
            Seamless PIL RGB image
        """
        if strength <= 0:
            return image

        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]
        blend_width = int(min(h, w) * 0.15 * strength)

        if blend_width < 4:
            return image

        # Step 1: Shift image by 50% - seams now in center
        shifted = np.roll(np.roll(img_array, h // 2, axis=0), w // 2, axis=1)

        # Step 2: Create smooth blend mask for center cross
        mask = self._create_center_blend_mask(h, w, blend_width)

        # Expand mask for RGB channels
        if len(img_array.shape) == 3:
            mask = mask[:, :, np.newaxis]

        # Step 3: Blend shifted with original at center
        result = shifted * (1 - mask) + img_array * mask

        # Step 4: Shift back
        result = np.roll(np.roll(result, -h // 2, axis=0), -w // 2, axis=1)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _create_center_blend_mask(self, h: int, w: int, blend_width: int) -> np.ndarray:
        """
        Create a mask that's 0 at center cross, 1 at edges.
        Uses smoothstep for artifact-free blending.
        """
        # Horizontal distance from center
        x = np.abs(np.arange(w) - w // 2)
        x_mask = self._smoothstep(x, blend_width // 2, blend_width)

        # Vertical distance from center
        y = np.abs(np.arange(h) - h // 2)
        y_mask = self._smoothstep(y, blend_width // 2, blend_width)

        # Combine: both must be away from center
        mask = np.outer(y_mask, np.ones(w)) * np.outer(np.ones(h), x_mask)

        return mask

    def _smoothstep(self, x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
        """Smooth Hermite interpolation between 0 and 1."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0, 1)
        return t * t * (3 - 2 * t)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_height_16bit(height: np.ndarray, path: str) -> None:
    """
    Save height map as 16-bit PNG for maximum precision.

    Args:
        height: Height map normalized [0, 1]
        path: Output file path
    """
    # Convert to 16-bit (0-65535)
    height_16bit = (height * 65535).astype(np.uint16)

    # OpenCV can write 16-bit PNGs
    cv2.imwrite(path, height_16bit)


def load_height_16bit(path: str) -> np.ndarray:
    """
    Load 16-bit height map and normalize to [0, 1].

    Args:
        path: Input file path

    Returns:
        Height map as float32 [0, 1]
    """
    height_16bit = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (height_16bit / 65535.0).astype(np.float32)
