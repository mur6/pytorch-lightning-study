import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.utils.cocoutils import bbox_from_binary_mask, calc_binary_mask_area, get_hand_mat_mask

BASE = Path("data/coco_keypoints/")
image_base = BASE / "train_images/"

keypoint_names = [
    "left-bottom",
    "right-bottom",
    "left-top",
    "right-top",
    "red[left-top]",
    "blue[left-bottom]",
    "brown[right-bottom",
    "pink[right-top]",
]

color_names = [
    "deeppink",
    "cyan",
    "coral",
    "green",
    "red",
    "blue",
    "brown",
    "fuchsia",
]


def view(anno, *, ax, jpeg_filename):
    im = Image.open(image_base / jpeg_filename)

    # create simple line plot
    ax.imshow(im)
    # ax.imshow(mask_img, alpha=0.6)
    kypt_list = anno["keypoints"]
    x, y, w, h = anno["bbox"]
    ax.add_patch(Rectangle((x, y), w, h, edgecolor="blue", fill=False, lw=2))
    # print(anno["keypoints"])
    for i, k_name, c_name in zip(range(0, len(kypt_list), 3), keypoint_names, color_names):
        x, y, visible = kypt_list[i : i + 3]
        if visible != 0:
            ax.plot(x, y, marker="o", color=c_name)
            ax.annotate(k_name, (x, y), color=c_name)


def load_coco(path):
    d = json.loads(path.read_text())
    return d["annotations"], d["images"]


def main(*, row, col):
    annotations, images = load_coco(BASE / "train_annotations.json")
    # jpeg_filename, png_filename = f"image_{num:06}.jpg", f"image_{num:06}.png"
    fig, axs = plt.subplots(row, col, figsize=(12, 7))
    fig.tight_layout()
    for i, ax in enumerate(axs.reshape(-1)):
        num = i + 1
        view(annotations[i], ax=ax, jpeg_filename=f"image_{num:06}.jpg")
    plt.savefig("aa.png")


parser = argparse.ArgumentParser()
parser.add_argument("--fig_row", type=int, default=2)
parser.add_argument("--fig_col", type=int, default=3)

if __name__ == "__main__":
    args = parser.parse_args()
    main(row=args.fig_row, col=args.fig_col)
