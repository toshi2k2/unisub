"""
Script to compute SubspaceAdapter components using Higher-Order SVD (HOSVD)
from pre-trained LoRA adapters.

Supports two tensor decomposition modes (--tucker_mode):

  mode-2  Concatenates each layer's LoRA matrices column-wise into a 2D
          matrix of shape (dim0, num_tasks * dim1) and extracts the left
          singular vectors via truncated SVD. Equivalent to standard PCA
          on the combined weight matrix.

  mode-3  Stacks each layer's LoRA matrices across tasks into a 3-mode
          tensor of shape (num_tasks, dim0, dim1) and applies Tucker
          decomposition. The mode-2 factor (dim0 × num_components) captures
          the shared row subspace across tasks.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tensorly as tl
from tensorly.decomposition import tucker
from peft import SubspaceAdapterConfig, get_peft_model, TaskType, load_peft_weights
from transformers import AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import save_file
from utils import (
    combine_loras,
    calculate_subspaceadapters,
    add_gs_vectors,
    add_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SubspaceAdapter components via HOSVD (Tucker decomposition)"
    )

    # Source LoRA adapters
    parser.add_argument(
        "--source_lora_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths or HuggingFace model IDs for source LoRA adapters",
    )
    parser.add_argument(
        "--source_lora_names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each source LoRA adapter (must match number of paths)",
    )

    # Target task configuration
    parser.add_argument(
        "--target_task_name",
        type=str,
        required=True,
        help="Name of the target task for SubspaceAdapter initialization",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels for the target task (default: 2)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-base",
        help="Base model name or path (default: roberta-base)",
    )
    parser.add_argument(
        "--subspaceadapter_r",
        type=int,
        default=8,
        help="LoRA rank for SubspaceAdapter (default: 8)",
    )

    # HOSVD configuration
    parser.add_argument(
        "--tucker_mode",
        type=int,
        default=3,
        choices=[2, 3],
        help="Tensor order for HOSVD: 2 = concatenated 2D SVD, 3 = 3D Tucker decomposition (default: 3)",
    )
    parser.add_argument(
        "--num_hosvd_components",
        type=int,
        default=32,
        help="Number of Tucker components to extract per layer (default: 32)",
    )
    parser.add_argument(
        "--num_gram_schmidt_components",
        type=int,
        default=32,
        help="Number of random Gram-Schmidt components to add (default: 32)",
    )
    parser.add_argument(
        "--loading_source_index",
        type=int,
        default=0,
        help="Index of source LoRA to use for computing loadings. Set to -1 for random loadings (default: 0)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for SubspaceAdapter adapter",
    )
    parser.add_argument(
        "--save_subspaceadapter_components",
        action="store_true",
        default=True,
        help="Save SubspaceAdapter components (default: True)",
    )
    parser.add_argument(
        "--save_subspaceadapter_loadings",
        action="store_true",
        default=True,
        help="Save SubspaceAdapter loadings (default: True)",
    )

    args = parser.parse_args()

    if len(args.source_lora_paths) != len(args.source_lora_names):
        parser.error(
            "Number of source_lora_paths must match number of source_lora_names"
        )

    args.total_components = args.num_hosvd_components + args.num_gram_schmidt_components
    print(f"Tucker mode:    {args.tucker_mode}")
    print(
        f"Total components: {args.total_components} "
        f"(HOSVD: {args.num_hosvd_components} + Gram-Schmidt: {args.num_gram_schmidt_components})"
    )
    return args


def load_source_loras(paths, names):
    """Load and combine source LoRA adapters into a nested dict."""
    lora_dict = {}
    lora_state_dicts = []
    for path, name in zip(paths, names):
        print(f"Loading LoRA adapter: {name} from {path}")
        state_dict = load_peft_weights(path)
        lora_state_dicts.append(state_dict)
        lora_dict = combine_loras(lora_dict, state_dict, name)
    return lora_dict, lora_state_dicts


def get_mode2_subspace(
    lora_dict: dict,
    num_components: int,
) -> dict:
    """
    Compute subspace bases via mode-2 decomposition (truncated SVD on a 2D matrix).

    For each layer, concatenates LoRA weight matrices from all source tasks
    column-wise into a 2D matrix of shape (dim0, num_tasks * dim1), where
    dim0 >= dim1 after orientation. The first `num_components` left singular
    vectors form the shared subspace basis.

    Args:
        lora_dict: Nested dict mapping layer keys to task-name → weight tensor.
        num_components: Number of left singular vectors to retain.

    Returns:
        Dict mapping each layer key to {"eigenvectors": Tensor(dim0, num_components)},
        compatible with calculate_subspaceadapters().
    """
    hosvd_dict = {}
    for layer_key, task_tensors in lora_dict.items():
        matrices = []
        for tensor in task_tensors.values():
            t = tensor.to(torch.float32)
            if t.shape[0] < t.shape[1]:
                t = t.t()
            matrices.append(t)

        # Concatenate column-wise: (dim0, num_tasks * dim1)
        mat_2d = torch.cat(matrices, dim=1)
        k = min(num_components, mat_2d.shape[0], mat_2d.shape[1])
        U, S, Vh = torch.linalg.svd(mat_2d, full_matrices=False)
        basis = U[:, :k].contiguous()
        hosvd_dict[layer_key] = {"eigenvectors": basis}

    return hosvd_dict


def get_mode3_subspace(
    lora_dict: dict,
    num_components: int,
) -> dict:
    """
    Compute subspace bases via mode-3 HOSVD (Tucker decomposition on a 3D tensor).

    For each layer, stacks LoRA weight matrices from all source tasks into a
    3-mode tensor of shape (num_tasks, dim0, dim1), where dim0 >= dim1 after
    orientation. Tucker decomposition extracts the mode-2 factor of shape
    (dim0, num_components) as the shared subspace basis.

    Args:
        lora_dict: Nested dict mapping layer keys to task-name → weight tensor.
        num_components: Number of Tucker components to retain along mode-2.

    Returns:
        Dict mapping each layer key to {"eigenvectors": Tensor(dim0, num_components)},
        compatible with calculate_subspaceadapters().
    """
    hosvd_dict = {}
    for layer_key, task_tensors in lora_dict.items():
        matrices = []
        for tensor in task_tensors.values():
            t = tensor.to(torch.float32)
            if t.shape[0] < t.shape[1]:
                t = t.t()
            matrices.append(t)

        # Stack into 3-mode tensor: (num_tasks, dim0, dim1)
        tensor_3d = torch.stack(matrices, dim=0)
        N, d0, d1 = tensor_3d.shape

        rank = [N, min(num_components, d0), d1]
        core, factors = tucker(
            tl.tensor(tensor_3d.numpy(), dtype=tl.float32),
            rank=rank,
            init="svd",
            tol=1e-6,
        )

        # factors[1] is the mode-2 factor: shape (d0, min(num_components, d0))
        basis = torch.tensor(factors[1], dtype=torch.float32).contiguous()
        hosvd_dict[layer_key] = {"eigenvectors": basis}

    return hosvd_dict


def compute_subspaceadapter(args, lora_dict, loading_source_sd):
    """Compute SubspaceAdapter components and loadings via HOSVD."""
    if args.tucker_mode == 2:
        print(
            f"Computing mode-2 subspace (SVD) with {args.num_hosvd_components} components..."
        )
        hosvd_dict = get_mode2_subspace(lora_dict, args.num_hosvd_components)
    else:
        print(
            f"Computing mode-3 subspace (Tucker) with {args.num_hosvd_components} components..."
        )
        hosvd_dict = get_mode3_subspace(lora_dict, args.num_hosvd_components)

    compute_loadings = args.loading_source_index >= 0
    if compute_loadings:
        print(f"Computing loadings from source LoRA index {args.loading_source_index}")
    else:
        print("Keeping loadings random (loading_source_index=-1)")

    subspaceadapter_sd = calculate_subspaceadapters(
        hosvd_dict,
        loading_source_sd,
        args.num_hosvd_components,
        compute_loadings,
    )

    print(
        f"Adding {args.num_gram_schmidt_components} Gram-Schmidt orthogonal vectors..."
    )
    subspaceadapter_sd = add_gs_vectors(
        subspaceadapter_sd, args.num_gram_schmidt_components
    )

    return subspaceadapter_sd


def save_subspaceadapter(args, subspaceadapter_sd):
    """Save SubspaceAdapter adapter config and weights to disk."""
    subspaceadapter_config = SubspaceAdapterConfig(
        r=args.subspaceadapter_r,
        num_components=args.total_components,
        task_type=TaskType.SEQ_CLS,
    )

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=model_config,
    )
    model = get_peft_model(model, subspaceadapter_config)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving SubspaceAdapter to {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        save_subspaceadapter_components=args.save_subspaceadapter_components,
        save_subspaceadapter_loadings=args.save_subspaceadapter_loadings,
    )

    adapter_path = os.path.join(args.output_dir, "adapter_model.safetensors")
    save_file(subspaceadapter_sd, adapter_path)
    print(f"Saved adapter weights to {adapter_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("SubspaceAdapter via HOSVD")
    print("=" * 60)
    print(f"Tucker mode:     {args.tucker_mode}")
    print(f"Target task:     {args.target_task_name}")
    print(f"Source LoRAs:    {args.source_lora_names}")
    print(f"Base model:      {args.model_name_or_path}")
    print(f"Output dir:      {args.output_dir}")
    print("=" * 60)

    lora_dict, lora_state_dicts = load_source_loras(
        args.source_lora_paths, args.source_lora_names
    )

    loading_source_sd = (
        lora_state_dicts[args.loading_source_index]
        if args.loading_source_index >= 0
        else None
    )

    subspaceadapter_sd = compute_subspaceadapter(args, lora_dict, loading_source_sd)
    save_subspaceadapter(args, subspaceadapter_sd)

    print("=" * 60)
    print("HOSVD SubspaceAdapter computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
