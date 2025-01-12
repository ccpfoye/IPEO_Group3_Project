import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor


def process_file(file_path: str) -> tuple:
    """
    Processes a file containing image and label data.
    -----
    The function expects the file to be in a binary format that can be loaded
    using `pickle.load`.
    """
    with open(file_path, "rb") as f:
        image, label = pickle.load(f)
    return image, label


def load_data_from_directory(directory: str):
    """
    Load images and labels from a specified directory.
    This function scans the given directory for files, processes each file in parallel,
    and returns the images and labels as numpy arrays.
    Parameters
    ----------
    directory : str
        The path to the directory containing the image files.
    Returns
    -------
    images : np.ndarray
        An array of images loaded from the directory.
    labels : np.ndarray
        An array of labels corresponding to the images.
    """

    images = []
    labels = []

    # Use os.scandir() for efficient directory traversal
    with os.scandir(directory) as entries:
        file_paths = [entry.path for entry in entries if entry.is_file()]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, file_paths))

    # Unpack results
    for image, label in results:
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


class ImageDataset(Dataset):
    """A custom Dataset class for loading image data and labels for PyTorch models.
    The dataset handles multi-band images stored as NumPy arrays and their corresponding labels.
    It supports optional transformations and band selection.
    Attributes:
        images (np.ndarray): Array of images with shape (N, H, W, C).
        labels (np.ndarray): Array of labels with shape (N,).
        transform (callable, optional): Optional transform to be applied on images.
        selected_bands (list, optional): Indices of bands to select from images.
    """

    def __init__(self, images, labels, transform=None, selected_bands=None):
        """
        images: NumPy array of shape (N, H, W, C)
        labels: NumPy array of shape (N,)
        transform: Optional transform or augmentation (e.g., torchvision transforms)
        selected_bands: List or tuple of band indices to select, e.g., [0, 2]
                        If None, use all bands.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve single image and label
        img = self.images[idx]
        label = self.labels[idx]

        # If specific bands are requested, slice them here
        if self.selected_bands is not None:
            # Make sure the selected bands are within the range of channels
            # For example, if self.images.shape[3] == 5, valid bands are [0..4]
            img = img[self.selected_bands, ...]

        # Convert to float32 for PyTorch
        img = img.astype(np.float32)

        # Reorder to (C, H, W) for PyTorch Tensors
        # img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        # Convert to torch Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # If you have any augmentations/transformations in torchvision style:
        # Usually they expect a PIL image or a torch.Tensor with shape (C, H, W).
        # If you're using torchvision.transforms, apply them here:
        if self.transform is not None:
            # Depending on your transform pipeline, you might need to convert
            # the tensor to a PIL image or keep it as a tensor.
            # For demonstration, assume the transform works on a Tensor:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor
