# Replicate Cog Deployment Notes

## Çalışır Hale Getirme Adımları

### 1. Token Test (Önce basit model ile)
```python
# predict.py - GPU'suz basit test
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Setup complete!")

    def predict(self, text: str = Input(default="hello")) -> str:
        return f"Echo: {text}"
```

```yaml
# cog.yaml - minimal
build:
  python_version: "3.11"
predict: "predict.py:Predictor"
```

### 2. Model Pre-download (ÖNEMLİ!)
HuggingFace modelleri runtime'da indirilemez. Build sırasında indirilmeli:

```yaml
# cog.yaml
build:
  run:
    - python -c "from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', variant='fp16')"
```

```python
# predict.py - local_files_only=True kullan
self.pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True  # <-- ÖNEMLİ
)
```

### 3. GitHub Actions Workflow
```yaml
runs-on: 4-core-runner  # özel runner

steps:
  - uses: actions/checkout@v4
  - uses: replicate/setup-cog@v2
    with:
      token: ${{ secrets.REPLICATE_API_TOKEN }}
  - run: |
      cog build -t r8.im/USERNAME/MODEL-NAME
      cog push r8.im/USERNAME/MODEL-NAME
```

### 4. GitHub Secrets
- `REPLICATE_API_TOKEN` - Replicate API token'ı repo secrets'a ekle

## Replicate Billing Notları

- Cold start SÜRESİ de faturalandırılır (GPU allocate edildiği andan itibaren)
- Hardware değişikliği push gerektirmez (dashboard'dan yapılır)
- Stuck prediction'lar "Starting" durumunda para yakmaz

### GPU Fiyatları (yaklaşık)
| GPU | $/sn | 8dk cold start |
|-----|------|----------------|
| T4 | $0.000225 | ~$0.11 |
| L40S | $0.001400 | ~$0.67 |
| A100 | $0.001150 | ~$0.55 |

## Cold Start Azaltma

1. **Idle timeout artır** - Settings > Scaling
2. **Min instances = 1** - Warm tutar ama 7/24 para yakar
3. **Daha küçük model** - SD 1.5, Segmind SSD-1B vs.

## Model Alternatifleri

| Model | Boyut | Cold Start |
|-------|-------|------------|
| SDXL Turbo | ~6.5GB | 5-8 dk |
| SD 1.5 | ~4GB | 2-3 dk |
| Segmind SSD-1B | ~2.5GB | 1-2 dk |
