# SDXL Style LoRAs — SubspaceAdapter

Experiments applying SubspaceAdapter to Stable Diffusion XL (SDXL) style LoRAs from the [KappaNeuro](https://huggingface.co/KappaNeuro) collection. LoRA adapter names are listed in `lora_names.json`.

---


## Step 1: Compute adapter weights

`get_subspaceadapter.py` loads source LoRA adapters from `lora_names.json`, computes principal components via eigendecomposition, and saves either **reconstructed LoRA weights** or **SubspaceAdapter components + loadings** for every target LoRA.

### Reconstruction (default)

```bash
python get_subspaceadapter.py \
    --source_lora_config lora_names.json \
    --num_eigenvector_components 32 \
    --output_type reconstruction \
    --output_dir ./output
```

### SubspaceAdapter output

```bash
python get_subspaceadapter.py \
    --source_lora_config lora_names.json \
    --num_eigenvector_components 32 \
    --output_type subspaceadapter \
    --output_dir ./output
```

### Separate source and target configs

```bash
python get_subspaceadapter.py \
    --source_lora_config lora_names.json \
    --target_lora_config lora_names.json \
    --num_eigenvector_components 32 \
    --output_type reconstruction \
    --output_dir ./output
```

### Process specific target LoRAs

```bash
python get_subspaceadapter.py \
    --source_lora_config lora_names.json \
    --target_lora_names "KappaNeuro/studio-ghibli-style" "KappaNeuro/ukiyo-e-art" \
    --num_eigenvector_components 32 \
    --output_type reconstruction \
    --output_dir ./output
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--source_lora_config` | `lora_names.json` | JSON file listing source LoRA adapter HF model IDs |
| `--target_lora_config` | `None` | JSON file listing target LoRAs (defaults to source config) |
| `--target_lora_names` | `None` | Explicit list of target HF model IDs (overrides config) |
| `--model_name_or_path` | `stabilityai/stable-diffusion-xl-base-1.0` | Base SDXL model |
| `--num_eigenvector_components` | `32` | Number of principal components |
| `--output_type` | `reconstruction` | `reconstruction` or `subspaceadapter` |
| `--output_dir` | `./output` | Output directory |
| `--output_filename` | `weights_sdxl.safetensors` | Filename for saved weights |
| `--torch_dtype` | `float16` | Model dtype: `float16`, `float32`, `bfloat16` |
| `--unwind_tensor` | `False` | Flatten weight matrices before PCA |

### Output layout

Each target LoRA produces a subfolder inside `--output_dir`:

```
output/
├── studio_ghibli_style_reconstruction/
│   └── weights_sdxl.safetensors
├── studio_ghibli_style_subspaceadapter/
│   └── weights_sdxl.safetensors
...
```

---

## Step 2: Generate images

`sdxl_inference.py` iterates over every LoRA listed in `lora_names.json`, loads the corresponding pre-computed adapter from disk, and saves one image per style.

### Reconstructed LoRA

```bash
python sdxl_inference.py \
    --lora_names_config lora_names.json \
    --adapter_dir ./output \
    --output_type reconstruction
```

### SubspaceAdapter

```bash
python sdxl_inference.py \
    --lora_names_config lora_names.json \
    --adapter_dir ./output \
    --output_type subspaceadapter
```

### Custom prompt

```bash
python sdxl_inference.py \
    --lora_names_config lora_names.json \
    --adapter_dir ./output \
    --output_type reconstruction \
    --prompt "a futuristic cityscape at night"
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--lora_names_config` | `lora_names.json` | JSON file listing LoRA HF model IDs |
| `--adapter_dir` | `./output` | Directory containing pre-computed adapter subfolders |
| `--output_type` | `reconstruction` | `reconstruction` or `subspaceadapter` |
| `--prompt` | `None` | Shared prompt for all styles (auto-derived from style name if omitted) |
| `--negative_prompt` | `None` | Negative prompt |
| `--model_name_or_path` | `stabilityai/stable-diffusion-xl-base-1.0` | Base SDXL model |
| `--num_inference_steps` | `30` | Denoising steps |
| `--guidance_scale` | `7.5` | Classifier-free guidance scale |
| `--lora_scale` | `1.0` | LoRA weight scale |
| `--seed` | `0` | Random seed |
| `--width` / `--height` | `1024` | Output image dimensions |
| `--torch_dtype` | `float16` | Model dtype |
| `--device` | `cuda` | Inference device |

Generated images are saved to `<adapter_dir>/images_<output_type>/<lora_name>.png`.

---

## Shell scripts

```bash
# Compute reconstruction weights for all LoRAs in lora_names.json
sh scripts/generation/get_subspaceadapter.sh

# Generate images with reconstructed LoRA
sh scripts/inference/infer_recons.sh

# Generate images with SubspaceAdapter
sh scripts/inference/infer_subspaceadapter.sh
```
