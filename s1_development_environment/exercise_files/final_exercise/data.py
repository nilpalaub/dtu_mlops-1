import torch
import glob
from pathlib import Path

base_path = Path("data") / "corruptmnist"

def mnist():
    """Return train and test dataloaders for MNIST."""
    train_images = [torch.load(train_file) for train_file in glob.glob(str(base_path / "train_images_*.pt"))]
    train_labels = [torch.load(train_file) for train_file in glob.glob(str(base_path / "train_target_*.pt"))]
    
    test_images = torch.load(str(base_path / "test_images.pt"))
    test_labels = torch.load(str(base_path / "test_target.pt"))

    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)

    # return torch dataset from tensors of train and test images and labels
    train = torch.utils.data.TensorDataset(train_images, train_labels)
    test = torch.utils.data.TensorDataset(test_images, test_labels)
    return train, test
