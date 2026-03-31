# LLaMA Spectral Analysis

For the newer alpha/scree follow-up analysis built on top of this directory, start with [`../Main_weight_analysis/llama/README.md`](../Main_weight_analysis/llama/README.md).

Spectral analysis and PCA for LLaMA model weights across multiple fine-tuned variants.

## Workflow

The analysis consists of three steps:

### Step 1: Download Models

Download LLaMA models from HuggingFace using the model list.

```bash
python download_models.py --model_name_file llama_models.txt --target_folder ./llama_models
```

**Arguments:**
- `--model_name_file, -f`: Path to file with model names (default: `llama_models.txt`)
- `--target_folder, -t`: Directory to save downloaded models (required)

---

### Step 2: Run PCA Analysis

Perform PCA on model weights layer by layer (linear layers only, excluding norm/lm_head).

```bash
python run_pca.py --model_directory ./llama_models --target_folder ./llama_plots
```

**Arguments:**
- `--model_directory, -d`: Directory containing downloaded models (required)
- `--target_folder, -t`: Output directory for PCA results and plots (required)
- `--previous_folder, -p`: (Optional) Directory with previous PCA data

---

### Step 3: Generate Plots

Generate variance plots from PCA results.

```bash
# Generate both average and layer-wise plots
python generate_plots.py --pca_dir ./llama_plots/pca --output_dir ./results --plot_type both

# Generate only average plot
python generate_plots.py --pca_dir ./llama_plots/pca --output_dir ./results --plot_type average

# Generate only layer-wise plots
python generate_plots.py --pca_dir ./llama_plots/pca --output_dir ./results --plot_type layerwise
```

**Arguments:**
- `--pca_dir, -p`: Directory containing PCA results (.npy files)
- `--output_dir, -o`: Output directory for plots
- `--plot_type, -t`: `average` | `layerwise` | `both` (default: `both`)
- `--num_components, -n`: Number of components to plot (default: 100)

**Output:**
- `average`: Creates `average_variance_plot.png` in output directory
- `layerwise`: Creates individual plots in `output_dir/layerwise/` directory
