from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from src.utils.cocoutils import bbox_from_binary_mask, calc_binary_mask_area, get_hand_mat_mask

image_base = Path("data/coco_keypoints/train_images/")
maks_base = Path("data/coco_keypoints/train_masks/")


def view(jpeg_filename, png_filename):
    im = Image.open(image_base / jpeg_filename)
    mask_img = Image.open(maks_base / png_filename)
    hand, mat = get_hand_mat_mask(mask_img)
    # plt.imshow(im)
    # plt.show()

    fig, ax = plt.subplots()

    # create simple line plot
    ax.imshow(im)
    ax.imshow(mask_img, alpha=0.6)

    # add rectangle to plot
    for target in (hand, mat):
        x, y, w, h = bbox_from_binary_mask(target)
        ax.add_patch(Rectangle((x, y), w, h, edgecolor="blue", fill=False, lw=2))
        area = calc_binary_mask_area(target)
        print(area)

    plt.show()


def main():
    num = 2
    jpeg_filename, png_filename = f"image_{num:06}.jpg", f"image_{num:06}.png"
    view(jpeg_filename, png_filename)


if __name__ == "__main__":
    main()
