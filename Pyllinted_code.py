"""
This module contains the code to download datasets, preprocess images and bounding boxes,
train an object detection model, and evaluate the model on the dataset.
"""

import urllib.request
import tarfile
import json
import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
except ImportError as e:
    raise ImportError("Torch library not found. Please install using 'pip install torch'.") from e

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV library not found. Please install using 'pip install opencv-python'.") from e

try:
    from albumentations.pytorch import ToTensorV2
    from albumentations import (
        Compose, HorizontalFlip, VerticalFlip, RandomResizedCrop, Normalize, BboxParams
    )
except ImportError as e:
    raise ImportError("Albumentations library not found. Please install using 'pip install albumentations'.") from e

try:
    from effdet import create_model
except ImportError as e:
    raise ImportError("EffDet library not found. Please install using 'pip install effdet'.") from e

try:
    from timm import create_model as timm_create_model
    CSPDARKNET53 = timm_create_model('cspdarknet53', pretrained=True)
    CSPDARKNET53.eval()
except ImportError:
    CSPDARKNET53 = None
    print("timm library not found. Please install using 'pip install timm'.")


class CocoDataset(Dataset):
    """Custom COCO dataset class for loading and processing data."""

    def __init__(self, dataset_path, annotation_file):
        self.data = self.load_coco_dataset(dataset_path, annotation_file)

    @staticmethod
    def load_coco_dataset(dataset_path, annotation_file):
        """Load COCO dataset from annotation file."""
        with open(annotation_file, 'r', encoding='utf-8') as file:
            annotations = json.load(file)

        images = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in images:
                images[image_id] = []
            images[image_id].append(ann['bbox'])

        dataset = []
        for image_id, bboxes in images.items():
            image_path = f"{dataset_path}/{image_id}.jpg"
            image = Image.open(image_path).convert("RGB")
            dataset.append((image, bboxes))

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, targets = self.data[idx]
        valid_targets = [bbox for bbox in targets if check_bbox(bbox)]
        if valid_targets:
            image, valid_targets = preprocess_data(image, valid_targets)
            return image, valid_targets
        return self.__getitem__((idx + 1) % len(self.data))


def check_bbox(bbox):
    """Check if the bounding box coordinates are valid."""
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_max > x_min and y_max > y_min


def normalize_bbox(bbox, rows, cols):
    """Normalize bounding box coordinates."""
    try:
        x_min, y_min, width, height = map(float, bbox)
        x_max, y_max = x_min + width, y_min + height

        if x_max <= x_min:
            raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")

        if x_max > cols:
            x_max = cols - 1
        if y_max > rows:
            y_max = rows - 1

        x_min, x_max = min(x_min, x_max) / cols, max(x_min, x_max) / cols
        y_min, y_max = min(y_min, y_max) / rows, max(y_min, y_max) / rows

        if x_max > x_min and y_max > y_min:
            return x_min, y_min, x_max, y_max
        return None
    except ValueError as e:
        print(f"Error normalizing bbox {bbox}: {e}")
        return None


def preprocess_data(image, targets):
    """Preprocess image and bounding boxes."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomResizedCrop(512, 512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=BboxParams(format='pascal_voc', label_fields=['category_ids']))

    augmented = transform(image=image, bboxes=targets, category_ids=[1] * len(targets))
    image = augmented['image']
    targets = augmented['bboxes']

    rows, cols = image.shape[1], image.shape[2]
    final_targets = [
        normalize_bbox(bbox, rows, cols) for bbox in targets if normalize_bbox(bbox, rows, cols)
    ]
    return image, final_targets


def pad_data(data):
    """Pad image and target data."""
    max_height = max(image.shape[1] for image, _ in data)
    max_width = max(image.shape[2] for image, _ in data)
    max_targets = max(len(targets) for _, targets in data)

    padded_data = []
    for image, targets in data:
        pad_height = max_height - image.shape[1]
        pad_width = max_width - image.shape[2]
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)),
                              mode='constant', constant_values=0)
        padded_targets = targets + [[0, 0, 0, 0]] * (max_targets - len(targets))
        padded_data.append((padded_image, padded_targets))

    padded_images = torch.tensor(np.array([image for image, _ in padded_data]), dtype=torch.float32)
    padded_targets = torch.tensor(np.array([targets for _, targets in padded_data]), dtype=torch.float32)

    return padded_images, padded_targets


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, targets = zip(*batch)
    return pad_data(list(zip(images, targets)))


def train_step(model, images, targets, optimizer, criterion):
    """Training step for a single batch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)
    losses = []

    for output in outputs:
        pred_boxes = output[0].view(-1, 4)
        target_size = pred_boxes.shape[0:]
        resized_targets = F.interpolate(targets.unsqueeze(0), size=target_size,
                                        mode='nearest').squeeze(0)

        # Computing loss
        loss = criterion(pred_boxes, resized_targets)
        losses.append(loss)

    # Combining losses
    loss = sum(losses)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, dataloader, criterion):
    """Evaluate the model on a test dataset."""
    model.eval()
    losses = []

    for images, targets in dataloader:
        images = images.float()
        targets = targets.float()

        # Forward pass
        outputs = model(images)
        batch_loss = 0

        for output in outputs:
            pred_boxes = output[0].view(-1, 4)
            target_size = pred_boxes.shape[0:]
            resized_targets = F.interpolate(targets.unsqueeze(0), size=target_size,
                                            mode='nearest').squeeze(0)
            batch_loss += criterion(pred_boxes, resized_targets)

        mean_loss = batch_loss / len(outputs)
        losses.append(mean_loss.item())

    return np.mean(losses)


def download_and_extract_dataset(url, extract_path):
    """Download and extract dataset."""
    dataset_tar = f"{extract_path}.tar.gz"
    urllib.request.urlretrieve(url, dataset_tar)
    with tarfile.open(dataset_tar, "r:gz") as tar:
        tar.extractall()


def load_model(pretrained=True):
    """Create EfficientDet model."""
    return create_model('tf_efficientdet_d0', pretrained=pretrained)


def main():
    """Main function to run the training and evaluation."""
    appliance_dataset_url = (
        "https://s3.us-west-2.amazonaws.com/testing.resources/datasets/"
        "mscoco-samples/appliance-dataset-5-tat-10.tar.gz"
    )
    food_dataset_url = (
        "https://s3.us-west-2.amazonaws.com/testing.resources/datasets/"
        "mscoco-samples/food-dataset-5-tat-10.tar.gz"
    )
    appliance_dataset_path = 'datasets/appliance'
    food_dataset_path = 'datasets/food'
    appliance_annotation_file = f"{appliance_dataset_path}/annotations.json"
    food_annotation_file = f"{food_dataset_path}/annotations.json"

    download_and_extract_dataset(appliance_dataset_url, appliance_dataset_path)
    download_and_extract_dataset(food_dataset_url, food_dataset_path)

    appliance_dataset = CocoDataset(appliance_dataset_path, appliance_annotation_file)
    food_dataset = CocoDataset(food_dataset_path, food_annotation_file)
    combined_dataset = ConcatDataset([appliance_dataset, food_dataset])

    train_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    for epoch in range(5):
        for images, targets in train_loader:
            images = images.float()
            targets = targets.float()
            loss = train_step(model, images, targets, optimizer, criterion)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

        eval_loss = evaluate_model(model, train_loader, criterion)
        print(f"Epoch {epoch + 1}, Evaluation Loss: {eval_loss}")


if __name__ == "__main__":
    main()
