# GPT-2 Spectral Analysis

Spectral analysis and PCA for GPT-2 model weights across multiple fine-tuned variants.

## Workflow

The analysis consists of three steps:

### Step 1: Download Models

Download GPT-2 models from HuggingFace using the model list.

```bash
python download_models.py --model_name_file gpt2.txt --target_folder ./gpt2_models
```

**Arguments:**
- `--model_name_file, -f`: Path to file with model names (default: `gpt2.txt`)
- `--target_folder, -t`: Directory to save downloaded models (required)

---

### Step 2: Run PCA Analysis

Perform PCA on model weights layer by layer (linear layers only, excluding ln/wte/wpe).

```bash
python run_pca.py --model_directory ./gpt2_models --target_folder ./gpt2_plots
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
python generate_plots.py --pca_dir ./gpt2_plots/pca --output_dir ./results --plot_type both

# Generate only average plot
python generate_plots.py --pca_dir ./gpt2_plots/pca --output_dir ./results --plot_type average

# Generate only layer-wise plots
python generate_plots.py --pca_dir ./gpt2_plots/pca --output_dir ./results --plot_type layerwise
```

**Arguments:**
- `--pca_dir, -p`: Directory containing PCA results (.npy files)
- `--output_dir, -o`: Output directory for plots
- `--plot_type, -t`: `average` | `layerwise` | `both` (default: `both`)
- `--num_components, -n`: Number of components to plot (default: 100)

**Output:**
- `average`: Creates `average_variance_plot.png` in output directory
- `layerwise`: Creates individual plots in `output_dir/layerwise/` directory
