# Download ViT models from Hugging Face

from transformers import AutoModelForImageClassification
import os
import argparse


def main():
    argparser = argparse.ArgumentParser(
        description="Download ViT models from HuggingFace"
    )
    argparser.add_argument(
        "--model_name_file",
        "-f",
        type=str,
        default="./vit.txt",
        help="Path to file containing model names (one per line)",
    )
    argparser.add_argument(
        "--target_folder",
        "-t",
        type=str,
        help="Folder to save the downloaded models",
        required=True,
    )
    args = argparser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.model_name_file):
        print(f"File {args.model_name_file} does not exist.")
        return

    os.makedirs(args.target_folder, exist_ok=True)

    # Initialize an empty list to store model names
    model_name_list = []

    # Open the file in read mode
    with open(args.model_name_file, "r") as file:
        # Read each line in the file
        for line in file:
            model_name_list.append(line.strip())

    for i, name in enumerate(model_name_list):
        # Load the model
        try:
            print(f"Loading model {i+1}/{len(model_name_list)}: {name}")
            if os.path.exists(os.path.join(args.target_folder, name.split("/")[-1])):
                print(f"Model {name} already exists in {args.target_folder}. Skipping.")
                continue
            model = AutoModelForImageClassification.from_pretrained(name)
            # Save the model to a directory
            model.save_pretrained(os.path.join(args.target_folder, name.split("/")[-1]))
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            continue


if __name__ == "__main__":
    main()
