# PBR Texture Generator

Generate seamless PBR (Physically Based Rendering) texture maps from text descriptions.

Powered by **Playground v2.5** - a high-quality aesthetic image model.

## Output

This model generates 5 texture maps from a single prompt:

| Output | Description |
|--------|-------------|
| **color.png** | Base color/albedo texture |
| **normal.png** | Surface detail for lighting |
| **roughness.png** | Surface smoothness (black=smooth, white=rough) |
| **ao.png** | Ambient occlusion for soft shadows |
| **grid.png** | 2x2 preview of all maps |

## Usage

```python
import replicate

output = replicate.run(
    "vantilator2000/pbr-playground",
    input={
        "prompt": "seamless red brick wall texture, weathered, detailed",
        "negative_prompt": "blurry, text, watermark",
        "resolution": 1024,
        "tiling_strength": 0.5,
        "num_steps": 25,
        "seed": 42
    }
)

# output[0] = color, output[1] = normal, output[2] = roughness, output[3] = ao, output[4] = grid
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | - | Text description of the texture |
| `negative_prompt` | "" | Things to avoid in the texture |
| `resolution` | 1024 | Output size (512 or 1024) |
| `tiling_strength` | 0.5 | Seamless tiling blend (0-1) |
| `num_steps` | 25 | Inference steps (1-50, higher=better quality) |
| `seed` | -1 | Random seed (-1 for random) |

## Example Prompts

- `seamless dark wood texture, oak, detailed grain`
- `seamless marble texture, white with grey veins`
- `seamless grass texture, top-down view, lawn`
- `seamless concrete texture, weathered, cracks`
- `seamless metal texture, brushed steel`
- `seamless fabric texture, blue denim, detailed`
- `seamless stone wall texture, medieval castle`
- `seamless leather texture, brown, worn`

## Tips

1. **Include "seamless"** in your prompt for better tiling
2. **Use "top-down view"** for floor/ground textures
3. **Increase `num_steps`** (30-40) for higher quality
4. **Set `tiling_strength`** to 0.7+ for perfect seamless edges
5. **Use negative_prompt** to avoid unwanted elements like "blurry, text, watermark"

## Use Cases

- Game development (Unity, Unreal Engine)
- 3D rendering (Blender, Maya, 3ds Max)
- Architectural visualization
- Product design
- Digital art

## Model

Powered by [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) - optimized for aesthetic quality and photorealism.
