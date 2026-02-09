# Mistral LoRA Spectral Analysis

Spectral analysis and PCA for Mistral LoRA adapter weights across multiple fine-tuned variants.

## Workflow

The analysis consists of three steps:

### Step 1: Download LoRAs

Download Mistral LoRA adapters from HuggingFace using the JSON model list.

```bash
python download_loras.py --model_list lora_list.json --output_dir ./mistral_loras
```

**Arguments:**
- `--model_list, -m`: Path to JSON file with LoRA names (default: `lora_list.json`)
- `--output_dir, -o`: Directory to save downloaded LoRAs (required)

---

### Step 2: Run PCA Analysis

Perform PCA on LoRA weights (lora_A and lora_B matrices only).

```bash
python run_pca.py --lora_directory ./mistral_loras --target_folder ./mistral_plots
```

**Arguments:**
- `--lora_directory, -d`: Directory containing downloaded LoRAs (required)
- `--target_folder, -t`: Output directory for PCA results and plots (required)
- `--previous_folder, -p`: (Optional) Directory with previous PCA data

---

### Step 3: Generate Plots

Generate variance plots from PCA results.

```bash
# Generate both average and layer-wise plots
python generate_plots.py --pca_dir ./mistral_plots/pca --output_dir ./results --plot_type both

# Generate only average plot
python generate_plots.py --pca_dir ./mistral_plots/pca --output_dir ./results --plot_type average

# Generate only layer-wise plots
python generate_plots.py --pca_dir ./mistral_plots/pca --output_dir ./results --plot_type layerwise
```

**Arguments:**
- `--pca_dir, -p`: Directory containing PCA results (.npy files)
- `--output_dir, -o`: Output directory for plots
- `--plot_type, -t`: `average` | `layerwise` | `both` (default: `both`)
- `--num_components, -n`: Number of components to plot (default: 100)

**Output:**
- `average`: Creates `average_variance_plot.png` in output directory
- `layerwise`: Creates individual plots in `output_dir/layerwise/` directory
