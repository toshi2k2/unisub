import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import wandb
import os
import argparse


# --- Data Loading Helper Function ---
def get_datasets(dataset_name, train_transform, val_transform, data_root="./data"):
    """
    Downloads and returns the specified dataset.
    Args:
        dataset_name (str): The name of the dataset.
        train_transform (callable): The transformation to apply to the training data.
        val_transform (callable): The transformation to apply to the validation data.
        data_root (str): The root directory for the data.
    Returns:
        tuple: A tuple containing (train_dataset, val_dataset, num_classes).
    """
    print(f"Loading dataset: {dataset_name}")

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=val_transform
        )
        num_classes = 10

    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=val_transform
        )
        num_classes = 100

    elif dataset_name == "pets":
        train_dataset = torchvision.datasets.OxfordIIITPet(
            root=data_root, split="trainval", download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.OxfordIIITPet(
            root=data_root, split="test", download=True, transform=val_transform
        )
        num_classes = 37

    elif dataset_name in ["caltech101", "eurosat"]:
        if dataset_name == "caltech101":
            num_classes = 101
            dataset_train = torchvision.datasets.Caltech101(
                root=data_root, download=True, transform=train_transform
            )
            dataset_val = torchvision.datasets.Caltech101(
                root=data_root, download=True, transform=val_transform
            )
        else:  # eurosat
            num_classes = 10
            dataset_train = torchvision.datasets.EuroSAT(
                root=data_root, download=True, transform=train_transform
            )
            dataset_val = torchvision.datasets.EuroSAT(
                root=data_root, download=True, transform=val_transform
            )

        total_size = len(dataset_train)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()

        train_dataset = Subset(dataset_train, indices[:train_size])
        val_dataset = Subset(dataset_val, indices[train_size:])

    elif dataset_name == "imagenet":
        candidate_roots = [data_root, os.path.join(data_root, "imagenet")]
        imagenet_root = None
        for root in candidate_roots:
            train_dir = os.path.join(root, "train")
            val_dir = os.path.join(root, "val")
            if os.path.isdir(train_dir) and os.path.isdir(val_dir):
                imagenet_root = root
                break

        if imagenet_root is None:
            raise ValueError(
                "ImageNet folder structure not found. Expected either "
                f"'{data_root}/train' and '{data_root}/val' or "
                f"'{data_root}/imagenet/train' and '{data_root}/imagenet/val'."
            )

        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_root, "train"),
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_root, "val"),
            transform=val_transform,
        )
        num_classes = len(train_dataset.classes)

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    return train_dataset, val_dataset, num_classes


# --- Main Training Function ---
def train_model(args):
    """
    Main function to run the training and validation process.
    """
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config
    torch.manual_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = config.device
    print(f"Using device: {device}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset, val_dataset, num_classes = get_datasets(
        config.dataset, train_transform, val_transform, data_root=config.data_root
    )
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=0
    )

    best_val_acc = 0.0
    for epoch in range(config.epochs):
        print(f"--- Epoch {epoch+1}/{config.epochs} ---")

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} Acc: {val_acc:.4f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)

            # --- FILENAME MODIFICATION ---
            # The filename now includes batch size (bs) and learning rate (lr)
            model_path = os.path.join(
                "models",
                f"best_model_bs{config.batch_size}_lr{config.learning_rate}_{wandb.run.id}.pth",
            )

            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact("best-model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print("New best model saved!")

        scheduler.step()

    print("--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    wandb.finish()


# --- Main Execution Block with Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on various image classification datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "pets", "caltech101", "eurosat", "imagenet"],
        help="Dataset to use for training (default: cifar100)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for datasets (default: ./data). For ImageNet, expects train/ and val/.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the data loaders (default: 128)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., 'cuda', 'cuda:0', or 'cpu'; default: 'cuda')",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vision-finetuning-demo",
        help="Name of the W&B project",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Your W&B entity (username or team name)",
    )

    args = parser.parse_args()
    train_model(args)
