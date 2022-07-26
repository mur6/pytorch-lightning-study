import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.utils.cocoutils import bbox_from_binary_mask, calc_binary_mask_area, get_hand_mat_mask

BASE = Path("data/coco_keypoints/")
image_base = BASE / "train_images/"
maks_base = BASE / "train_masks/"


def view(anno, *, jpeg_filename):
    im = Image.open(image_base / jpeg_filename)

    fig, ax = plt.subplots()

    # create simple line plot
    ax.imshow(im)
    # ax.imshow(mask_img, alpha=0.6)
    kypt_list = anno["keypoints"]
    x, y, w, h = anno["bbox"]
    ax.add_patch(Rectangle((x, y), w, h, edgecolor="blue", fill=False, lw=2))
    # print(anno["keypoints"])
    for i in range(0, len(kypt_list), 3):
        x, y, visible = kypt_list[i : i + 3]
        if visible != 0:
            ax.plot(x, y, marker="o", color="red")
    plt.show()


def load_coco(path):
    d = json.loads(path.read_text())
    return d["annotations"], d["images"]


def main():
    annotations, images = load_coco(BASE / "train_annotations.json")
    # for image in images:
    #     print(image)
    # for anno in annotations:
    #     print(anno)

    # jpeg_filename, png_filename = f"image_{num:06}.jpg", f"image_{num:06}.png"
    for i in range(20):
        num = i + 1
        view(annotations[i], jpeg_filename=f"image_{num:06}.jpg")


if __name__ == "__main__":
    main()
