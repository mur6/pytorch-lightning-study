# 4. Detect objects in a few images!
import pickle

import matplotlib.pyplot as plt
from flash import DataKeys, Trainer
from flash.image import KeypointDetectionData
from matplotlib.patches import Rectangle
from PIL import Image


def get_predict_files(base_dir):
    iter = base_dir.glob("*.jpeg")
    iter = map(str, sorted(iter))
    print(iter)
    return list(iter)


def main(model, predict_files):
    datamodule = KeypointDetectionData.from_files(
        predict_files=predict_files,
        batch_size=4,
    )
    trainer = Trainer()
    predictions = trainer.predict(model, datamodule=datamodule)
    # predictions = trainer.predict(model, datamodule=datamodule, output="fiftyone")
    # session = visualize(predictions, wait=True)
    return predictions[0]


def view(filepath, *, bbox, keypoint):
    im = Image.open(filepath)
    im = im.resize((128, 128))
    fig, ax = plt.subplots()

    # create simple line plot
    ax.imshow(im)
    # ax.imshow(mask_img, alpha=0.6)
    x, y, w, h = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]
    ax.add_patch(Rectangle((x, y), w, h, edgecolor="blue", fill=False, lw=2))

    for item in keypoint:
        x, y = item["x"], item["y"]
        # if visible != 0:
        ax.plot(x, y, marker="o", color="red")
    plt.show()


if __name__ == "__main__":
    item_list = []
    with open("predictions.pkl", "rb") as f:
        d_list = pickle.load(f)
        for d in d_list:
            item_list += d

    for item in item_list:
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
        view(input_meta["filepath"], bbox=bbox, keypoint=keypoint)
        print()
