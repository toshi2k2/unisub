"""Spectral Analysis for GPT2 models"""

from transformers import AutoModelForCausalLM
import numpy as np
import torch
from sklearn.decomposition import PCA
import argparse
import os

import matplotlib.pyplot as plt

init_n_components = 300


def main(args):
    print("Total number of models: ", len(os.listdir(args.model_directory)))

    os.makedirs(os.path.join(args.target_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "cumul"), exist_ok=True)
    os.makedirs(os.path.join(args.target_folder, "pca"), exist_ok=True)

    if not os.path.exists(os.path.join(args.target_folder, "all_layerdata.npy")):

        layers = []
        all_dirs = os.listdir(args.model_directory)

        print(all_dirs)
        assert len(all_dirs) > 0

        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(args.model_directory, all_dirs[0])
        )
        all_data = {}

        # for k, _ in model.state_dict().items():
        #     print(k)
        # exit(0)

        for k, _ in model.state_dict().items():
            if (
                k.endswith(".weight")
                and "ln" not in k
                and "wte" not in k
                and "wpe" not in k
            ):
                if not os.path.exists(
                    os.path.join(args.target_folder, "pca", f"{k}_components.npy")
                ):
                    layers.append(k)
                    all_data[k] = []
        if len(layers) == 0:
            raise ValueError("No layers found in the model state dict.")
        print("Layers found: ", layers)

        # for layer in layers:
        #     if os.path.exists(os.path.join(args.target_folder, 'pca', f"{layer}_components.npy")):
        #         print(f"Skipping layer {layer} as PCA components already exist.")
        #         continue
        #     print(f"Processing layer: {layer}")
        #     data_list = []
        for model_name in all_dirs:
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(args.model_directory, model_name)
            )
            # try:
            # data_list.append(model.state_dict()[layer].cpu().numpy())
            for layer in layers:
                try:
                    all_data[layer].append(model.state_dict()[layer].cpu().numpy())
                except KeyError:
                    print(
                        f"KeyError: {layer} not found in model {model_name}. Skipping this model."
                    )
                    continue
            # print(f"Number of models loaded for layer {layer}: ", len(data_list))
        # for layer, data in all_data.items():
        #     if len(data) == 0:
        #         print(f"No data loaded for layer {layer}. Skipping PCA.")
        #         layers.remove(layer)
        #         all_data.pop(layer)

        del model
        np.save(os.path.join(args.target_folder, "all_layerdata.npy"), all_data)
    else:
        all_data = np.load(
            os.path.join(args.target_folder, "all_layerdata.npy"), allow_pickle=True
        ).item()
        print("Loaded all layer data from file.")
    for layer, data_list in all_data.items():
        print(f"Number of models loaded for layer {layer}: ", len(data_list))
        if len(data_list) == 0:
            print(f"No data loaded for layer {layer}. Skipping PCA.")
            layers.remove(layer)
            continue
        print(len(data_list))

        data_mean = sum(data_list) / len(data_list)
        data_list = [data - data_mean for data in data_list]
        data_array = np.concatenate(data_list, axis=0)
        del data_list

        # Perform PCA
        # if 'mlp' in layer:
        #     pca = PCA(n_components=900)
        # else:
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
            pca = PCA(n_components=300)
            try:
                pca.fit_transform(data_array)
                # found_pca = True
            except ValueError as e:
                print(f"ValueError: {e}. Skipping PCA for layer {layer}.")
                continue
            e_variance = pca.explained_variance_ratio_.sum()
            num_components = pca.n_components_
        if e_variance < 0.90:
            print(
                f"Explained variance {e_variance} is less than 0.90. Increasing n_components."
            )
            pca.n_components = int((num_components / e_variance) * 0.9)
            pca.n_components = min(pca.n_components, data_array.shape[0])
            print(f"New n_components: {pca.n_components}")
            pca = PCA(n_components=pca.n_components)
            try:
                pca.fit_transform(data_array)
            except ValueError as e:
                print(f"ValueError: {e}. Skipping PCA for layer {layer}.")
                # found_pca = False
                pca = PCA(n_components=300)
                pca.fit_transform(data_array)
                continue
            if pca.n_components >= data_array.shape[0]:
                print(
                    f"n_components {pca.n_components} exceeds number of samples {data_array.shape[0]}. Stopping PCA."
                )
                continue
            e_variance = pca.explained_variance_ratio_.sum()
            print(f"New explained variance: {e_variance}")
        # pca = PCA()

        print("Sum of explained variance: ", sum(pca.explained_variance_ratio_))

        # Plot the PCA result
        plt.figure(figsize=(12, 6))
        PC_values = np.arange(pca.n_components_) + 1
        # plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
        plt.bar(
            PC_values, pca.explained_variance_ratio_, color="skyblue", edgecolor="blue"
        )
        plt.title(layer)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        plt.savefig(os.path.join(args.target_folder, "plots", f"{layer}.png"), dpi=200)
        plt.close()
        print(f"Saved PCA plot for {layer} to {args.target_folder}")

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(PC_values, cumulative_variance, "b-", linewidth=2)
        plt.title(layer)
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Proportion of Variance Explained")
        plt.grid()
        plt.savefig(os.path.join(args.target_folder, "cumul", f"{layer}.png"), dpi=200)
        plt.close()

        # Save PCA components and explained variance
        np.save(
            os.path.join(args.target_folder, "pca", f"{layer}_components.npy"),
            pca.components_,
        )
        np.save(
            os.path.join(args.target_folder, "pca", f"{layer}_explained_variance.npy"),
            pca.explained_variance_ratio_,
        )
        print(
            f"Saved PCA components and explained variance for {layer} to {args.target_folder}"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run PCA on GPT-2 model weights")
    argparser.add_argument(
        "--model_directory",
        "-d",
        type=str,
        required=True,
        help="Directory containing downloaded GPT-2 models",
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
