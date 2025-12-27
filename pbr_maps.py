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
        Make height map seamlessly tileable using edge blending.

        Creates seamless edges by blending opposite borders together.
        Uses cosine interpolation for smooth falloff.

        Args:
            height: Height map [0, 1]
            strength: Blend region size (0-1, as fraction of image)

        Returns:
            Tile-safe height map
        """
        if strength <= 0:
            return height

        h, w = height.shape
        # Blend region - larger = smoother but more blurred edges
        blend_size = max(4, int(min(h, w) * 0.20 * strength))

        result = height.copy()

        # Create cosine weights for smooth blending
        weights = self._cosine_weights(blend_size)

        # Horizontal seamless: blend left edge with wrapped right edge
        for i in range(blend_size):
            t = weights[i]
            # Left side: blend with right side
            result[:, i] = (1 - t) * height[:, w - blend_size + i] + t * height[:, i]
            # Right side: blend with left side (mirror)
            result[:, w - 1 - i] = (1 - t) * height[:, blend_size - 1 - i] + t * height[:, w - 1 - i]

        # Vertical seamless: blend top edge with wrapped bottom edge
        for i in range(blend_size):
            t = weights[i]
            # Top side: blend with bottom side
            result[i, :] = (1 - t) * result[h - blend_size + i, :] + t * result[i, :]
            # Bottom side: blend with top side (mirror)
            result[h - 1 - i, :] = (1 - t) * result[blend_size - 1 - i, :] + t * result[h - 1 - i, :]

        return result.astype(np.float32)

    def _cosine_weights(self, size: int) -> np.ndarray:
        """Generate cosine interpolation weights from 0 to 1."""
        t = np.linspace(0, np.pi / 2, size)
        return np.sin(t)  # Smooth 0 to 1 curve


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
# EMISSIVE MAP GENERATION
# =============================================================================

class EmissiveGenerator:
    """
    Generate emissive/glow maps by detecting bright, saturated areas.
    Perfect for neon, LED, glowing elements in textures.
    """

    def estimate_emissive(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        saturation_boost: bool = True
    ) -> Image.Image:
        """
        Extract emissive areas from RGB image.

        Detects pixels that are both bright AND saturated (neon colors).
        Pure white is less emissive than bright blue/pink/green.

        Args:
            image: RGB PIL image
            threshold: Brightness threshold for emission (0-1)
            saturation_boost: Boost saturated colors (neon effect)

        Returns:
            PIL grayscale emissive map (white = glowing)
        """
        # Convert to HSV using OpenCV (fast)
        rgb = np.array(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Extract channels (OpenCV HSV: H=0-179, S=0-255, V=0-255)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0  # Brightness

        # Emissive = bright areas
        brightness_mask = np.clip((value - threshold) / (1 - threshold + 1e-8), 0, 1)

        if saturation_boost:
            # Boost saturated bright colors (neon effect)
            # Saturated + bright = very emissive
            # Desaturated + bright (white) = less emissive
            sat_factor = 0.5 + 0.5 * saturation  # 0.5 to 1.0
            emissive = brightness_mask * sat_factor
        else:
            emissive = brightness_mask

        # Smooth slightly for nicer glow
        emissive = gaussian_filter(emissive, sigma=1.0)

        # Enhance contrast
        emissive = np.clip(emissive * 1.5, 0, 1)

        return Image.fromarray((emissive * 255).astype(np.uint8), mode="L")


# =============================================================================
# SEAMLESS TILING UTILITIES
# =============================================================================

class SeamlessTiling:
    """
    Make RGB images seamlessly tileable using edge blending.
    """

    def make_seamless(
        self,
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """
        Make RGB image seamlessly tileable using edge blending.

        Creates seamless edges by blending opposite borders together.
        Uses cosine interpolation for smooth falloff.

        Args:
            image: RGB PIL image
            strength: Blend region size (0-1, as fraction of image)

        Returns:
            Seamless PIL RGB image
        """
        if strength <= 0:
            return image

        img = np.array(image, dtype=np.float32)
        h, w = img.shape[:2]
        blend_size = max(4, int(min(h, w) * 0.20 * strength))

        result = img.copy()

        # Create cosine weights for smooth blending
        weights = self._cosine_weights(blend_size)

        # Horizontal seamless: blend left edge with wrapped right edge
        for i in range(blend_size):
            t = weights[i]
            result[:, i] = (1 - t) * img[:, w - blend_size + i] + t * img[:, i]
            result[:, w - 1 - i] = (1 - t) * img[:, blend_size - 1 - i] + t * img[:, w - 1 - i]

        # Vertical seamless: blend top edge with wrapped bottom edge
        for i in range(blend_size):
            t = weights[i]
            result[i, :] = (1 - t) * result[h - blend_size + i, :] + t * result[i, :]
            result[h - 1 - i, :] = (1 - t) * result[blend_size - 1 - i, :] + t * result[h - 1 - i, :]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _cosine_weights(self, size: int) -> np.ndarray:
        """Generate cosine interpolation weights from 0 to 1."""
        t = np.linspace(0, np.pi / 2, size)
        return np.sin(t)


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
