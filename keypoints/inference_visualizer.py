# 4. Detect objects in a few images!
import pickle

import matplotlib.pyplot as plt
from flash import DataKeys
from matplotlib.patches import Rectangle
from PIL import Image

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


def view(filepath, *, ax, bbox, keypoint):
    im = Image.open(filepath)
    im = im.resize((128, 128))
    # create simple line plot
    ax.imshow(im)
    # ax.imshow(mask_img, alpha=0.6)
    x, y, w, h = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]
    ax.add_patch(Rectangle((x, y), w, h, edgecolor="blue", fill=False, lw=2))

    for item, k_name, c_name in zip(keypoint, keypoint_names, color_names):
        x, y = item["x"], item["y"]
        # if visible != 0:
        # ax.plot(x, y, marker="o", color="red")
        ax.plot(x, y, marker="o", color=c_name)
        ax.annotate(k_name, (x, y), color=c_name)


if __name__ == "__main__":
    item_list = []
    with open("data/outputs/predictions.pkl", "rb") as f:
        d_list = pickle.load(f)
        for d in d_list:
            item_list += d
    row, col = 2, 3
    fig, axs = plt.subplots(row, col, figsize=(12, 7))
    fig.tight_layout()
    for item, ax in zip(item_list[:6], axs.reshape(-1)):
        # pred_info = item[DataKeys.METADATA]
        # print(f"pred_info: {pred_info}")
        meta = item[DataKeys.METADATA]
        print(f"meta: {meta}")
        scores = item[DataKeys.PREDS]["scores"][0]
        bbox = item[DataKeys.PREDS]["bboxes"][0]
        keypoint = item[DataKeys.PREDS]["keypoints"][0]

        print(f"scores={scores}")
        input_meta = item[DataKeys.INPUT][DataKeys.METADATA]
        print(f"input={input_meta['filepath']}")
        view(input_meta["filepath"], ax=ax, bbox=bbox, keypoint=keypoint)
        print()
    plt.savefig("b2.png")
