"""Spectral Analysis for LLaMA models"""

from transformers import AutoModelForCausalLM
import numpy as np
import torch
from sklearn.decomposition import PCA
import argparse
import os
import matplotlib.pyplot as plt

init_n_components = 300


def main(args):
    print(f"Looking for models in: {args.model_directory}")
    all_dirs = os.listdir(args.model_directory)
    print(f"Total number of models: {len(all_dirs)}")

    os.makedirs(os.path.join(args.target_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "cumul"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "pca"), exist_ok=True)

    if not os.path.exists(os.path.join(args.target_folder, "all_layerdata.npy")):
        assert len(all_dirs) > 0, "No model directories found"

        # Load first model to get layer names
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(args.model_directory, all_dirs[0])
        )

        layers = []
        all_data = {}

        for k, _ in model.state_dict().items():
            # Filter for linear layers only (exclude norm and lm_head)
            if (
                "layers." in k
                and k.endswith(".weight")
                and "norm" not in k
                and "lm_head" not in k
            ):
                if not os.path.exists(
                    os.path.join(args.target_folder, "pca", f"{k}_components.npy")
                ):
                    layers.append(k)
                    all_data[k] = []

        if len(layers) == 0:
            raise ValueError("No layers found in the model state dict.")
        print(f"Layers found: {layers}")

        # Load weights from all models
        for model_name in all_dirs:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(args.model_directory, model_name)
                )
                for layer in layers:
                    try:
                        all_data[layer].append(model.state_dict()[layer].cpu().numpy())
                    except KeyError:
                        print(
                            f"KeyError: {layer} not found in model {model_name}. Skipping."
                        )
                        continue
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue

        del model
        np.save(os.path.join(args.target_folder, "all_layerdata.npy"), all_data)
    else:
        all_data = np.load(
            os.path.join(args.target_folder, "all_layerdata.npy"), allow_pickle=True
        ).item()
        print("Loaded all layer data from file.")
        layers = list(all_data.keys())

    for layer, data_list in all_data.items():
        print(f"Number of models loaded for layer {layer}: {len(data_list)}")
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
    argparser = argparse.ArgumentParser(description="Run PCA on LLaMA model weights")
    argparser.add_argument(
        "--model_directory",
        "-d",
        type=str,
        required=True,
        help="Directory containing downloaded LLaMA models",
    )
    argparser.add_argument(
        "--target_folder",
        "-t",
        type=str,
        required=True,
        help="Folder to save PCA results and plots",
    )
    argparser.add_argument(
        "--previous_folder",
        "-p",
        type=str,
        default=None,
        help="Folder where previous PCA data is stored (optional)",
    )
    args = argparser.parse_args()
    main(args)
