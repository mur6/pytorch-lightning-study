import argparse
from pathlib import Path

import flash

# from flash.core.utilities.imports import example_requires
from flash.image import KeypointDetectionData, KeypointDetector


def main(*, data_dir, batch_size, max_epochs):
    # data_dir = "data/coco_keypoints"
    train_folder = data_dir / "train_images"
    train_ann_file = data_dir / "train_annotations.json"
    datamodule = KeypointDetectionData.from_coco(
        train_folder=train_folder,
        train_ann_file=train_ann_file,
        val_split=0.1,
        transform_kwargs=dict(image_size=(128, 128)),
        batch_size=batch_size,
    )
    print(f"num of dataset: {len(datamodule.train_dataset)}")
    print(f"num_classes: {datamodule.num_classes}")
    # 2. Build the task
    backbone = "resnet34_fpn"
    model = KeypointDetector(
        head="keypoint_rcnn",
        backbone=backbone,
        num_keypoints=8,
        num_classes=2,
        # num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=max_epochs)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 5. Save the model!
    trainer.save_checkpoint(f"models/kypt_{backbone}_ba{batch_size}_ep{max_epochs}.pt")


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, required=True)
parser.add_argument("--batch_size", type=int, default=4, required=True)
parser.add_argument("--max_epochs", type=int, default=10, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(data_dir=args.data_dir, batch_size=args.batch_size, max_epochs=args.max_epochs)
