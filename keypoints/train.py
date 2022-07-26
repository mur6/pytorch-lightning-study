import sys
from pathlib import Path

import flash

# from flash.core.utilities.imports import example_requires
from flash.image import KeypointDetectionData, KeypointDetector


def main(data_dir):
    # data_dir = "data/coco_keypoints"
    train_folder = data_dir / "train_images"
    train_ann_file = data_dir / "train_annotations.json"
    datamodule = KeypointDetectionData.from_coco(
        train_folder=train_folder,
        train_ann_file=train_ann_file,
        val_split=0.1,
        # transform_kwargs=dict(image_size=(128, 128)),
        batch_size=2,
    )
    print(f"num of dataset: {len(datamodule.train_dataset)}")
    print(f"num_classes: {datamodule.num_classes}")
    # 2. Build the task
    model = KeypointDetector(
        head="keypoint_rcnn",
        backbone="resnet34_fpn",
        num_keypoints=8,
        num_classes=2,
        # num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=3)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 5. Save the model!
    trainer.save_checkpoint("models/keypoint_detection_model.pt")


if __name__ == "__main__":
    p = Path(sys.argv[1])
    main(p)
