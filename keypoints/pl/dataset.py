from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def xyxy2torchvision(coords: torch.tensor):
    new_coords = torch.empty(size=[coords.size()[0], 3])

    new_coords[:, 0] = coords[:, 0]
    new_coords[:, 1] = coords[:, 1]
    new_coords[:, 2] = 1

    return new_coords


class MatDataset(Dataset):
    def __init__(self, root_folder: Path, transform=None):
        super().__init__()
        self.transform = transform

        if self.flag == "train":
            self.images = np.load(file=str(root_folder / "xtrain.npy"))
            self.labels = np.load(file=str(root_folder / "ytrain.npy"))
        else:
            self.images = np.load(file=str(root_folder / "xtest.npy"))

    def __getitem__(self, item):
        x = self.images[item]

        # https://discuss.pytorch.org/t/normalizing-images-between-0-and-1-with-totensor-doesnt-work/104363
        if self.transform:
            x = self.transform(x)
        else:
            transforms.ToTensor()(x) / 255.0

        # Format of y is (num_instances, num_keypoints_in_instance, 3)
        # 3 means that each keypoint should be represented in this form: [x, y, visibility],
        # where visibility = 0 means that keypoint is not visible (visible - 1)
        y = self.labels[item]
        y = torch.as_tensor(y * 256.0).reshape(-1, 2)

        boxes = torch.as_tensor([torch.min(y[:, 0]), torch.min(y[:, 1]), torch.max(y[:, 0]), torch.max(y[:, 1])])

        points = xyxy2torchvision(y)
        output = {
            "boxes": boxes.unsqueeze(0),
            "labels": torch.as_tensor(1).unsqueeze(0),
            "keypoints": points.unsqueeze(0),
        }
        return x, output

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root_dir = Path("data/")
    train_dataset = MatDataset(root_dir=root_dir)
    image, labels = train_dataset[0]

    image = image.permute(1, 2, 0).numpy()
    labels = labels["keypoints"].squeeze()[:, :2]

    plt.imshow(image)
    plt.scatter(labels[:, 0], labels[:, 1], c="r")
    plt.show()
