import sys

import flash
from flash.core.utilities.imports import example_requires
from flash.image import KeypointDetectionData, KeypointDetector


# datamodule = KeypointDetectionData.from_coco(
#             ...     train_folder="train_folder",
#             ...     train_ann_file="train_annotations.json",
#             ...     predict_folder="predict_folder",
#             ...     transform_kwargs=dict(image_size=(128, 128)),
#             ...     batch_size=2,
#             ... )
def main():
    datamodule = KeypointDetectionData.from_coco(
        train_folder="data/coco_keypoints",
        val_split=0.1,
        transform_kwargs=dict(image_size=(128, 128)),
        batch_size=4,
    )

    # 2. Build the task
    model = KeypointDetector(
        head="keypoint_rcnn",
        backbone="resnet18_fpn",
        num_keypoints=1,
        num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 5. Save the model!
    trainer.save_checkpoint("keypoint_detection_model.pt")


if __name__ == "__main__":
    main(sys.argv[1])
