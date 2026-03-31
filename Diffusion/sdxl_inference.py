"""
Script for SDXL inference using LoRA or SubspaceAdapter adapters.
This script loads a Stable Diffusion XL model and generates images
using specified LoRA weights.
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import DiffusionPipeline


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
        description="Generate images using SDXL with LoRA or SubspaceAdapter adapters"
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model name or path (default: stabilityai/stable-diffusion-xl-base-1.0)",
    )

    # Batch configuration from JSON
    parser.add_argument(
        "--lora_names_config",
        type=str,
        default="lora_names.json",
        help="JSON file mapping indices to HuggingFace model IDs (default: lora_names.json)",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./output",
        help="Base directory where SubspaceAdapter/reconstruction outputs were saved (default: ./output)",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="reconstruction",
        choices=["subspaceadapter", "reconstruction"],
        help="Which adapter type to load: 'subspaceadapter' or 'reconstruction' (default: reconstruction)",
    )

    # LoRA scale
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0)",
    )

    # Generation configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for image generation. If not provided, a prompt is auto-derived from each LoRA name.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for model loading (default: float16)",
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


def load_base_pipeline(args):
    """Load the base SDXL pipeline (without LoRA)."""
    print(f"Loading SDXL pipeline from {args.model_name_or_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
    ).to(args.device)
    return pipe


def load_lora_into_pipeline(
    pipe, lora_path, weight_name=None, adapter_name="default", use_subspaceadapter=False
):
    """Load LoRA (or SubspaceAdapter) weights into an existing pipeline."""
    print(f"Loading adapter weights from {lora_path}...")
    load_kwargs = {
        "adapter_name": adapter_name,
        "use_subspaceadapter": use_subspaceadapter,
    }
    if weight_name:
        load_kwargs["weight_name"] = weight_name
    pipe.load_lora_weights(lora_path, **load_kwargs)
    return pipe


def generate_image(pipe, args, prompt=None):
    """Generate an image using the pipeline."""
    prompt = prompt or args.prompt
    print(f"Generating image with prompt: '{prompt}'")

    generator = torch.manual_seed(args.seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        cross_attention_kwargs={"scale": args.lora_scale},
        generator=generator,
    ).images[0]

    return image


def generate_batch_from_json(args):
    """Generate images for all LoRAs listed in lora_names_config."""
    loras = load_loras_from_json(args.lora_names_config)
    print(f"Batch generation: {len(loras)} LoRAs from {args.lora_names_config}")
    print(f"Output type: {args.output_type}")

    use_subspaceadapter = args.output_type == "subspaceadapter"
    suffix = f"_{args.output_type}"
    output_dir = os.path.join(args.adapter_dir, f"images_{args.output_type}")
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_base_pipeline(args)

    for lora_path, lora_name in loras:
        adapter_folder = os.path.join(args.adapter_dir, f"{lora_name}{suffix}")
        if not os.path.isdir(adapter_folder):
            print(f"Skipping {lora_name}: adapter folder not found at {adapter_folder}")
            continue

        # Unload any previously loaded LoRA before loading the next one
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        load_lora_into_pipeline(
            pipe,
            adapter_folder,
            weight_name="weights_sdxl.safetensors",
            adapter_name=lora_name,
            use_subspaceadapter=use_subspaceadapter,
        )

        # Build prompt: use provided prompt or auto-derive from LoRA name
        style_label = lora_name.replace("_style", "").replace("_", " ")
        prompt = (
            args.prompt if args.prompt else f"a beautiful image in {style_label} style"
        )

        image = generate_image(pipe, args, prompt=prompt)

        out_file = os.path.join(output_dir, f"{lora_name}.png")
        image.save(out_file)
        print(f"Saved image to {out_file}")

    print("=" * 60)
    print(f"Batch generation complete! Images saved to {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("SDXL Inference with LoRA/SubspaceAdapter")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Config: {args.lora_names_config}")
    print(f"Output type: {args.output_type}")
    print("=" * 60)

    generate_batch_from_json(args)


if __name__ == "__main__":
    main()
