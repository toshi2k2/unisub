"""
Script to compute SubspaceAdapter components from pre-trained SDXL LoRA adapters.
This script computes eigenvectors from source LoRA adapters and generates
SubspaceAdapter initialization for target styles in Stable Diffusion XL.
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import save_file
from utils import (
    consolidate_loras_sdxl,
    get_eigenvectors,
    calculate_reconstructed_loras,
    calculate_subspaceadapters,
)


def lora_path_to_name(lora_path: str) -> str:
    """Derive a short name from a HuggingFace model ID or local path."""
    return lora_path.split("/")[-1].replace("-", "_")


def load_loras_from_json(json_path: str):
    """Load LoRA (path, name) pairs from a JSON file mapping indices to HF model IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [
        (v, lora_path_to_name(v))
        for _, v in sorted(data.items(), key=lambda x: int(x[0]))
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SubspaceAdapter components from pre-trained SDXL LoRA adapters"
    )

    # Source LoRA adapters
    parser.add_argument(
        "--source_lora_config",
        type=str,
        default="lora_names.json",
        help="JSON file mapping indices to HuggingFace model IDs for source LoRAs (default: lora_names.json)",
    )

    # Target LoRA configuration
    parser.add_argument(
        "--target_lora_config",
        type=str,
        default=None,
        help="JSON file mapping indices to HuggingFace model IDs for target LoRAs. "
        "If not provided, uses source_lora_config.",
    )
    parser.add_argument(
        "--target_lora_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific target LoRA HuggingFace paths to process (overrides target_lora_config)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model name or path (default: stabilityai/stable-diffusion-xl-base-1.0)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading (default: float16)",
    )
    parser.add_argument(
        "--num_eigenvector_components",
        type=int,
        default=32,
        help="Number of eigenvector components to use (default: 32)",
    )

    # Processing options
    parser.add_argument(
        "--unwind_tensor",
        action="store_true",
        help="Unwind tensors when computing eigenvectors",
    )

    # Output configuration
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["reconstruction", "subspaceadapter"],
        default="reconstruction",
        help="Output type: 'reconstruction' for reconstructed LoRA weights, "
        "'subspaceadapter' for SubspaceAdapter components+loadings (default: reconstruction)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for adapter weights (default: ./output)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="weights_sdxl.safetensors",
        help="Output filename for weights (default: weights_sdxl.safetensors)",
    )

    args = parser.parse_args()

    # Set torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_map[args.torch_dtype]

    return args


def load_pipeline(model_name_or_path, torch_dtype):
    """Load the SDXL pipeline."""
    print(f"Loading SDXL pipeline from {model_name_or_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )
    return pipe


def load_source_loras(pipe, source_loras):
    """Load and combine source LoRA adapters."""
    lora_dict = {}

    for path, name in source_loras:
        print(f"Loading LoRA adapter: {name} from {path}")
        lora_dict = consolidate_loras_sdxl(pipe, lora_dict, path, name)

    return lora_dict


def compute_and_save_subspaceadapter(args, pipe, eig_dict):
    """Compute SubspaceAdapter and save to disk."""
    print(f"Computing SubspaceAdapter for {args.target_lora_name}...")
    subspaceadapter_sd = calculate_subspaceadapters(
        pipe, args.target_lora_path, eig_dict, args.num_eigenvector_components
    )

    # Create output directory
    subspaceadapter_dir = os.path.join(
        args.output_dir, f"{args.target_lora_name}_subspaceadapter"
    )
    os.makedirs(subspaceadapter_dir, exist_ok=True)

    output_path = os.path.join(subspaceadapter_dir, args.output_filename)
    save_file(subspaceadapter_sd, output_path)
    print(f"Saved SubspaceAdapter weights to {output_path}")


def compute_and_save_reconstruction(args, pipe, eig_dict):
    """Compute reconstructed LoRA and save to disk."""
    print(f"Computing reconstructed LoRA for {args.target_lora_name}...")
    recons_sd = calculate_reconstructed_loras(
        pipe, args.target_lora_path, eig_dict, args.num_eigenvector_components
    )

    # Create output directory
    recons_dir = os.path.join(args.output_dir, f"{args.target_lora_name}_recons")
    os.makedirs(recons_dir, exist_ok=True)

    # Save reconstructed LoRA weights
    output_path = os.path.join(recons_dir, args.output_filename)
    save_file(recons_sd, output_path)
    print(f"Saved reconstructed LoRA weights to {output_path}")


def process_single_target(args, pipe, eig_dict, target_lora_path, target_lora_name):
    """Compute SubspaceAdapter or reconstruction for a single target LoRA."""
    args.target_lora_path = target_lora_path
    args.target_lora_name = target_lora_name
    if args.output_type == "subspaceadapter":
        compute_and_save_subspaceadapter(args, pipe, eig_dict)
    else:
        compute_and_save_reconstruction(args, pipe, eig_dict)


def main():
    args = parse_args()

    print("=" * 60)
    print("SubspaceAdapter Component Computation for SDXL")
    print("=" * 60)
    print(f"Source config: {args.source_lora_config}")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Components: {args.num_eigenvector_components}")
    print(f"Output type: {args.output_type}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load pipeline
    pipe = load_pipeline(args.model_name_or_path, args.torch_dtype)

    # Load source LoRAs from JSON and compute eigenvectors
    print(f"Loading source LoRAs from {args.source_lora_config}...")
    source_loras = load_loras_from_json(args.source_lora_config)
    print(f"Source LoRAs: {len(source_loras)} adapters")
    lora_dict = load_source_loras(pipe, source_loras)

    print("Computing eigenvectors...")
    eig_dict = get_eigenvectors(lora_dict, args.unwind_tensor)

    # Determine target LoRAs
    if args.target_lora_names:
        target_loras = [(p, lora_path_to_name(p)) for p in args.target_lora_names]
    elif args.target_lora_config:
        target_loras = load_loras_from_json(args.target_lora_config)
    else:
        target_loras = load_loras_from_json(args.source_lora_config)

    print(f"Target LoRAs: {len(target_loras)} adapters")
    os.makedirs(args.output_dir, exist_ok=True)

    for target_path, target_name in target_loras:
        print(f"\nProcessing target: {target_name} ({target_path})")
        process_single_target(args, pipe, eig_dict, target_path, target_name)

    print("=" * 60)
    print("SubspaceAdapter computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
