import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm


def get_matrix_reconstruction(model_dict, component_dict):
    reconstructed = {}
    for key in tqdm(component_dict.keys(), desc="Reconstructing layers"):
        if key not in model_dict:
            continue
        components = component_dict[key]
        layer = model_dict[key].to(torch.float64)
        layer_shape = layer.shape
        layer_flat = layer.view(layer.size(0), -1)
        loadings = torch.mm(components.t(), layer_flat)
        recons = torch.mm(components, loadings).view(layer_shape).contiguous()
        reconstructed[key] = recons

    for key, value in model_dict.items():
        reconstructed.setdefault(key, value)

    if set(reconstructed.keys()) != set(model_dict.keys()):
        missing = set(model_dict.keys()) - set(reconstructed.keys())
        raise RuntimeError(f"Reconstruction failed for keys: {sorted(missing)}")
    return reconstructed


def combine_loras(model_dict, state_dict, key_name):
    for key, value in state_dict.items():
        if (
            key.endswith(".weight")
            and "bn" not in key
            and "fc" not in key
            and len(value.shape) >= 2
        ):
            flattened = value.view(value.size(0), -1)
            model_dict.setdefault(key, {})
            model_dict[key][key_name] = flattened
    return model_dict


def get_eigenvectors(lora_dict, rank=64, niter=50):
    eigen_dict = {}
    for layer_key in tqdm(lora_dict.keys(), desc="Computing eigenspaces"):
        tensor_list = [t for t in lora_dict[layer_key].values()]
        concat_tensors = torch.cat(tensor_list, dim=1).to(torch.float64)
        q = min(rank, min(concat_tensors.shape))
        u, _, _ = torch.svd_lowrank(concat_tensors, q=q, niter=niter)
        eigen_dict[layer_key] = u.contiguous()
    return eigen_dict


def parse_model_list(model_paths_arg):
    entries = [item.strip() for item in model_paths_arg.split(",") if item.strip()]
    parsed = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Invalid model spec '{entry}'. Expected format 'name:/abs/or/rel/path.pt'."
            )
        name, path = entry.split(":", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def run_compute_eig(model_specs, eig_out, rank, niter):
    model_dict = {}
    for name, model_path in model_specs:
        print(f"Loading model for eigenspace: {name} <- {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model_dict = combine_loras(model_dict, state_dict, name)
    eigs = get_eigenvectors(model_dict, rank=rank, niter=niter)
    torch.save(eigs, eig_out)
    print(f"Saved eigenspaces to: {eig_out}")


def run_reconstruct(model_specs, eig_path, out_dir, suffix):
    print(f"Loading eigenspaces from: {eig_path}")
    eig = torch.load(eig_path, map_location="cpu")
    os.makedirs(out_dir, exist_ok=True)
    for name, model_path in model_specs:
        print(f"Reconstructing model: {name} <- {model_path}")
        model = torch.load(model_path, map_location="cpu")
        new_sd = get_matrix_reconstruction(model, eig)
        out_path = os.path.join(out_dir, f"{name}{suffix}.pt")
        torch.save(new_sd, out_path)
        print(f"Saved reconstructed model: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute shared layer eigenspaces and/or reconstruct CNN checkpoints "
            "using those eigenspaces."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="reconstruct",
        choices=["compute-eig", "reconstruct", "all"],
        help="Workflow mode: compute eigenspaces, reconstruct, or both.",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        default=(
            "cifar10:/mnt/d/JHU/final_cnn/cifar10.pt,"
            "cifar100:/mnt/d/JHU/final_cnn/cifar100.pt,"
            "eurosat:/mnt/d/JHU/final_cnn/eurosat.pt,"
            "pets:/mnt/d/JHU/final_cnn/pets.pt,"
            "imagenet:/mnt/d/JHU/final_cnn/imagenet.pt"
        ),
        help="Comma-separated model specs in the format 'name:path'.",
    )
    parser.add_argument(
        "--eig_path",
        type=str,
        default="64_resnet50.pt",
        help="Path to eigenspace .pt file (input for reconstruct, output for compute-eig).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/d/JHU/final_cnn",
        help="Directory to write reconstructed checkpoints.",
    )
    parser.add_argument(
        "--recon_suffix",
        type=str,
        default="_64_rc",
        help="Suffix appended to reconstructed output filenames.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Target rank (q) for svd_lowrank.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=50,
        help="Number of power iterations for svd_lowrank.",
    )
    args = parser.parse_args()

    model_specs = parse_model_list(args.model_paths)
    eig_path = str(Path(args.eig_path))

    if args.mode in {"compute-eig", "all"}:
        run_compute_eig(
            model_specs=model_specs,
            eig_out=eig_path,
            rank=args.rank,
            niter=args.niter,
        )

    if args.mode in {"reconstruct", "all"}:
        run_reconstruct(
            model_specs=model_specs,
            eig_path=eig_path,
            out_dir=args.out_dir,
            suffix=args.recon_suffix,
        )


if __name__ == "__main__":
    main()
