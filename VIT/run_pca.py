"""Spectral Analysis for ViT models - use llmadapt conda env"""

from transformers import AutoModelForImageClassification
import numpy as np
import torch
from sklearn.decomposition import PCA
import argparse
import os
import matplotlib.pyplot as plt


def main(args):
    print(f"Looking for models in: {args.model_directory}")
    all_dirs = os.listdir(args.model_directory)
    print(f"Total number of models: {len(all_dirs)}")

    os.makedirs(os.path.join(args.target_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "cumul"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "pca"), exist_ok=True)

    assert len(all_dirs) > 0, "No model directories found"

    # Load first model to get layer names
    model = AutoModelForImageClassification.from_pretrained(
        os.path.join(args.model_directory, all_dirs[0])
    )

    layers = []
    for k, _ in model.state_dict().items():
        # Filter for linear layers only (exclude norm and classifier)
        if (
            "layer." in k
            and k.endswith(".weight")
            and "norm" not in k
            and "classifier" not in k
        ):
            layers.append(k)

    if len(layers) == 0:
        raise ValueError("No layers found in the model state dict.")
    print(f"Layers found: {layers}")

    for layer in layers:
        if os.path.exists(
            os.path.join(args.target_folder, "pca", f"{layer}_components.npy")
        ):
            print(f"Skipping layer {layer} as PCA components already exist.")
            continue

        data_list = []
        for model_name in all_dirs:
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    os.path.join(args.model_directory, model_name)
                )
                data_list.append(model.state_dict()[layer].cpu().numpy())
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue

        if len(data_list) == 0:
            print(f"No data loaded for layer {layer}. Skipping.")
            continue

        print(f"Number of models loaded for layer {layer}: {len(data_list)}")

        data_mean = sum(data_list) / len(data_list)
        data_list = [data - data_mean for data in data_list]
        data_array = np.concatenate(data_list, axis=0)
        del data_list

        # Perform PCA
        n_components = min(300, data_array.shape[0] - 1, data_array.shape[1] - 1)
        pca = PCA(n_components=n_components)
        try:
            pca.fit_transform(data_array)
        except ValueError as e:
            print(f"ValueError: {e}. Skipping PCA for layer {layer}.")
            continue

        print(f"Sum of explained variance: {sum(pca.explained_variance_ratio_)}")

        # Plot the PCA result
        plt.figure(figsize=(12, 6))
        PC_values = np.arange(pca.n_components_) + 1
        plt.bar(
            PC_values, pca.explained_variance_ratio_, color="skyblue", edgecolor="blue"
        )
        plt.title(layer)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        safe_layer = layer.replace(".", "_").replace("/", "_")
        plt.savefig(
            os.path.join(args.target_folder, "plots", f"{safe_layer}.png"), dpi=200
        )
        plt.close()
        print(f"Saved PCA plot for {layer}")

        # Cumulative variance plot
        plt.figure(figsize=(12, 6))
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(PC_values, cumulative_variance, "b-", linewidth=2)
        plt.title(layer)
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Proportion of Variance Explained")
        plt.grid()
        plt.savefig(
            os.path.join(args.target_folder, "cumul", f"{safe_layer}.png"), dpi=200
        )
        plt.close()

        # Save PCA components and explained variance
        np.save(
            os.path.join(args.target_folder, "pca", f"{safe_layer}_components.npy"),
            pca.components_,
        )
        np.save(
            os.path.join(
                args.target_folder, "pca", f"{safe_layer}_explained_variance.npy"
            ),
            pca.explained_variance_ratio_,
        )
        print(f"Saved PCA components and explained variance for {layer}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run PCA on ViT model weights")
    argparser.add_argument(
        "--model_directory",
        "-d",
        type=str,
        required=True,
        help="Directory containing downloaded ViT models",
    )
    argparser.add_argument(
        "--target_folder",
        "-t",
        type=str,
        required=True,
        help="Folder to save PCA results and plots",
    )
    args = argparser.parse_args()
    main(args)
