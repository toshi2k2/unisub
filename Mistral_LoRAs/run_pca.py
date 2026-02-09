"""Spectral Analysis for Mistral LoRA adapters"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from safetensors import safe_open
import argparse
import os
import matplotlib.pyplot as plt

init_n_components = 300


def load_lora_weights(lora_path: str) -> dict:
    """Load LoRA weights from a safetensors file."""
    safetensor_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(safetensor_file):
        # Try alternative name
        safetensor_file = os.path.join(lora_path, "adapter_model.bin")
        if os.path.exists(safetensor_file):
            return torch.load(safetensor_file, map_location="cpu")
        raise FileNotFoundError(f"No adapter file found in {lora_path}")

    weights = {}
    with safe_open(safetensor_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def main(args):
    print(f"Looking for LoRAs in: {args.lora_directory}")
    all_dirs = [
        d
        for d in os.listdir(args.lora_directory)
        if os.path.isdir(os.path.join(args.lora_directory, d))
    ]
    print(f"Total number of LoRAs: {len(all_dirs)}")

    os.makedirs(os.path.join(args.target_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "cumul"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "pca"), exist_ok=True)

    if not os.path.exists(os.path.join(args.target_folder, "all_layerdata.npy")):
        assert len(all_dirs) > 0, "No LoRA directories found"

        # Load first LoRA to get layer names
        first_lora_path = os.path.join(args.lora_directory, all_dirs[0])
        first_weights = load_lora_weights(first_lora_path)

        layers = []
        all_data = {}

        # Filter for linear layer weights only (lora_A and lora_B)
        for k in first_weights.keys():
            if "lora_A" in k or "lora_B" in k:
                if not os.path.exists(
                    os.path.join(args.target_folder, "pca", f"{k}_components.npy")
                ):
                    layers.append(k)
                    all_data[k] = []

        if len(layers) == 0:
            raise ValueError("No LoRA layers found.")
        print(f"Layers found: {layers}")

        # Load weights from all LoRAs
        for lora_name in all_dirs:
            lora_path = os.path.join(args.lora_directory, lora_name)
            try:
                weights = load_lora_weights(lora_path)
                for layer in layers:
                    if layer in weights:
                        all_data[layer].append(weights[layer].cpu().numpy())
                    else:
                        print(f"Layer {layer} not found in {lora_name}. Skipping.")
            except Exception as e:
                print(f"Error loading {lora_name}: {e}")
                continue

        np.save(os.path.join(args.target_folder, "all_layerdata.npy"), all_data)
    else:
        all_data = np.load(
            os.path.join(args.target_folder, "all_layerdata.npy"), allow_pickle=True
        ).item()
        print("Loaded all layer data from file.")
        layers = list(all_data.keys())

    for layer, data_list in all_data.items():
        print(f"Number of LoRAs loaded for layer {layer}: {len(data_list)}")
        if len(data_list) == 0:
            print(f"No data loaded for layer {layer}. Skipping PCA.")
            continue

        data_mean = sum(data_list) / len(data_list)
        data_list = [data - data_mean for data in data_list]
        data_array = np.concatenate(data_list, axis=0)
        del data_list

        # Perform PCA
        if args.previous_folder and os.path.exists(
            os.path.join(args.previous_folder, "pca", f"{layer}_components.npy")
        ):
            explained_variance = np.load(
                os.path.join(
                    args.previous_folder, "pca", f"{layer}_explained_variance.npy"
                )
            )
            e_variance = explained_variance.sum()
            num_components = explained_variance.shape[0]
            print(f"Loaded previous PCA components for {layer}.")
        else:
            n_components = min(300, data_array.shape[0] - 1, data_array.shape[1] - 1)
            pca = PCA(n_components=n_components)
            try:
                pca.fit_transform(data_array)
            except ValueError as e:
                print(f"ValueError: {e}. Skipping PCA for layer {layer}.")
                continue
            e_variance = pca.explained_variance_ratio_.sum()
            num_components = pca.n_components_

        if e_variance < 0.90:
            print(
                f"Explained variance {e_variance} is less than 0.90. Increasing n_components."
            )
            new_n_components = min(
                int((num_components / e_variance) * 0.9), data_array.shape[0] - 1
            )
            pca = PCA(n_components=new_n_components)
            try:
                pca.fit_transform(data_array)
            except ValueError as e:
                print(f"ValueError: {e}. Skipping PCA for layer {layer}.")
                continue
            e_variance = pca.explained_variance_ratio_.sum()
            print(f"New explained variance: {e_variance}")

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

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(12, 6))
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
    parser = argparse.ArgumentParser(description="Run PCA on Mistral LoRA weights")
    parser.add_argument(
        "--lora_directory",
        "-d",
        type=str,
        required=True,
        help="Directory containing downloaded LoRAs",
    )
    parser.add_argument(
        "--target_folder",
        "-t",
        type=str,
        required=True,
        help="Folder to save PCA results and plots",
    )
    parser.add_argument(
        "--previous_folder",
        "-p",
        type=str,
        default=None,
        help="Folder where previous PCA data is stored (optional)",
    )
    args = parser.parse_args()
    main(args)
