"""Generate plots from PCA results for ViT models."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_explained_variance_files(pca_dir: str) -> dict:
    """Load all explained variance numpy files from the PCA directory."""
    files = os.listdir(pca_dir)
    data = {}

    for file in files:
        if file.endswith(".npy") and "explained_variance" in file:
            layer_name = file.replace("_explained_variance.npy", "")
            data[layer_name] = np.load(os.path.join(pca_dir, file))

    return data


def pad_arrays(data: dict) -> np.ndarray:
    """Pad all arrays to the same length and return as numpy array."""
    arrays = list(data.values())
    max_length = max(arr.shape[0] for arr in arrays)

    padded_data = [
        np.pad(
            arr, (0, max_length - arr.shape[0]), mode="constant", constant_values=0.0
        )
        for arr in arrays
    ]

    return np.array(padded_data)


def generate_average_plot(data: dict, output_dir: str, num_components: int = 100):
    """Generate average/combined plot across all layers."""
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["axes.titleweight"] = "bold"
    padded_data = pad_arrays(data)
    mean_variance = padded_data.mean(axis=0)[:num_components]
    std_variance = padded_data.std(axis=0)[:num_components]

    plt.figure(figsize=(14, 7))
    components = np.arange(1, len(mean_variance) + 1)
    colors = plt.cm.viridis(mean_variance / mean_variance.max())

    plt.bar(
        components,
        mean_variance,
        yerr=std_variance,
        capsize=6,
        error_kw={"elinewidth": 1.5, "alpha": 0.8},
        color=colors,
        edgecolor="blue",
        label="Mean Explained Variance",
    )

    # Add title in the middle of the figure
    plt.text(
        0.55,
        0.8,
        f"Mean Eigenvalue/Variance Plot of {len(data)} ViT Model Layers",
        fontsize=25,
        ha="center",
        va="center",
        transform=plt.gcf().transFigure,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="blue"),
    )

    plt.xlabel("Principal Component", fontsize=14)
    plt.ylabel("Explained Variance", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "average_variance_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved average plot to {output_path}")


def generate_layerwise_plots(data: dict, output_dir: str, num_components: int = 200):
    """Generate individual plots for each layer."""
    layerwise_dir = os.path.join(output_dir, "layerwise")
    os.makedirs(layerwise_dir, exist_ok=True)

    for layer_name, variance in data.items():
        plt.figure(figsize=(12, 6))

        variance_to_plot = variance[:num_components]
        components = np.arange(1, len(variance_to_plot) + 1)
        colors = plt.cm.viridis(variance_to_plot / variance_to_plot.max())

        plt.bar(components, variance_to_plot, color=colors, edgecolor="blue")

        plt.title(f"Explained Variance for {layer_name}", fontsize=14)
        plt.xlabel("Principal Component", fontsize=12)
        plt.ylabel("Explained Variance", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Create safe filename
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        output_path = os.path.join(layerwise_dir, f"{safe_name}.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(data)} layer-wise plots to {layerwise_dir}")


def main(args):
    """Main function to generate plots."""
    print(f"Loading PCA results from {args.pca_dir}")
    data = load_explained_variance_files(args.pca_dir)
    print(f"Found {len(data)} layers with explained variance data")

    if len(data) == 0:
        raise ValueError(
            "No explained variance files found in the specified directory."
        )

    if args.plot_type in ("average", "both"):
        print("Generating average plot...")
        generate_average_plot(data, args.output_dir, args.num_components)

    if args.plot_type in ("layerwise", "both"):
        print("Generating layer-wise plots...")
        generate_layerwise_plots(data, args.output_dir, args.num_components)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from PCA results")
    parser.add_argument(
        "--pca_dir",
        "-p",
        type=str,
        required=True,
        help="Directory containing PCA results (explained variance .npy files)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--plot_type",
        "-t",
        type=str,
        choices=["average", "layerwise", "both"],
        default="both",
        help="Type of plots to generate: average, layerwise, or both (default: both)",
    )
    parser.add_argument(
        "--num_components",
        "-n",
        type=int,
        default=100,
        help="Number of principal components to plot (default: 100)",
    )
    args = parser.parse_args()
    main(args)
