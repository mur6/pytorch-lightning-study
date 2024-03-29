import argparse
import enum
from pathlib import Path

import flash
import torch

# from flash.core.utilities.imports import example_requires
from flash.image import KeypointDetectionData, KeypointDetector

AVAILABLE_BACKBONES = [
    "resnet101_fpn",
    "resnet152_fpn",
    "resnet18_fpn",
    "resnet34_fpn",
    "resnet50_fpn",
    "resnext101_32x8d_fpn",
    "resnext50_32x4d_fpn",
    "wide_resnet101_2_fpn",
    "wide_resnet50_2_fpn",
]


class Strategy(enum.Enum):
    freeze = "freeze"
    freeze_unfreeze = ("freeze_unfreeze", 70)
    unfreeze_milestones = ("unfreeze_milestones", ((35, 35), 2))

    @classmethod
    def get_all(cls):
        return tuple(cls.__members__.keys())


def main(*, data_dir, backbone, strategy, batch_size, max_epochs):
    train_folder = data_dir / "train_images"
    train_ann_file = data_dir / "train_annotations.json"
    datamodule = KeypointDetectionData.from_coco(
        train_folder=train_folder,
        train_ann_file=train_ann_file,
        val_split=0.1,
        transform_kwargs=dict(image_size=(128, 128)),
        batch_size=batch_size,
    )
    print(f"backbone: {backbone}\tstrategy: {strategy}")
    print(f"num of dataset: {len(datamodule.train_dataset)}")
    print(f"num_classes: {datamodule.num_classes}")
    print(f"Gpu count: {torch.cuda.device_count()}")
    model = KeypointDetector(
        head="keypoint_rcnn",
        backbone=backbone,
        num_keypoints=8,
        num_classes=2,
        # num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=max_epochs, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule, strategy=Strategy[strategy].value)

    # 5. Save the model!
    trainer.save_checkpoint(f"models/kypt_{backbone}_{strategy}_ba{batch_size}_ep{max_epochs}.pt")


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, required=True)
parser.add_argument("--backbone", choices=AVAILABLE_BACKBONES, required=True)
parser.add_argument("--strategy", choices=Strategy.get_all(), default=Strategy.unfreeze_milestones, required=True)
parser.add_argument("--batch_size", type=int, default=4, required=True)
parser.add_argument("--max_epochs", type=int, default=10, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        backbone=args.backbone,
        strategy=args.strategy,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
