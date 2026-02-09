"""Download Mistral LoRA adapters from HuggingFace based on a JSON model list."""

from huggingface_hub import snapshot_download
import json
import argparse
import os


def load_model_list(model_list_path: str) -> list:
    """Load model names from a JSON file."""
    with open(model_list_path, "r") as f:
        data = json.load(f)
    # Handle dict format (like {"0": "model_name", "1": "model_name2", ...})
    if isinstance(data, dict):
        models = list(data.values())
    else:
        models = data
    return models


def download_lora(model_name: str, output_dir: str) -> bool:
    """Download a single LoRA adapter and save it to the output directory."""
    # Create a safe directory name by replacing / with _
    safe_name = model_name.replace("/", "_")
    model_path = os.path.join(output_dir, safe_name)

    if os.path.exists(model_path):
        print(f"LoRA {model_name} already exists at {model_path}. Skipping.")
        return True

    try:
        print(f"Downloading LoRA: {model_name}")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"Saved LoRA to {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False


def main(args):
    """Main function to download all LoRAs."""
    # Load model list
    models = load_model_list(args.model_list)
    print(f"Found {len(models)} LoRAs in {args.model_list}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download each LoRA
    success_count = 0
    fail_count = 0

    for i, model_name in enumerate(models):
        print(f"\n[{i + 1}/{len(models)}] Processing {model_name}")
        if download_lora(model_name, args.output_dir):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"Download complete: {success_count} succeeded, {fail_count} failed")
    print(f"LoRAs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Mistral LoRAs from HuggingFace"
    )
    parser.add_argument(
        "--model_list",
        "-m",
        type=str,
        default="lora_list.json",
        help="Path to the JSON file containing LoRA names",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save downloaded LoRAs",
    )
    args = parser.parse_args()
    main(args)
