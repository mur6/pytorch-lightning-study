# 4. Detect objects in a few images!
import argparse
import pickle
import pprint
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from flash import Trainer
from flash.image import KeypointDetectionData, KeypointDetector
from PIL import Image
from torch import nn


class WrapperModel(nn.Module):
    def __init__(self, in_model):
        super().__init__()
        self.model = in_model

    def forward(self, input):
        result = self.model(input)
        r = result[0]
        # pprint.pprint(result)
        boxes = r["boxes"]
        labels = r["labels"]
        scores = r["scores"]
        keypoints = r["keypoints"]
        # rest_outputs = r[""]
        # if len(boxes) > 0:
        #     return boxes[0], labels[0], scores[0], keypoints[0]
        # return [{"boxes": boxes[0], "labels": labels[0], "scores": scores[0], "keypoints": keypoints[0]}]
        # return {"boxes": boxes[0], "labels": labels[0], "scores": scores[0], "keypoints": keypoints[0]}
        return boxes[0], labels[0], scores[0], keypoints[0]


def make_wrapper_model():
    model_file = "models/kypt_resnet18_fpn_ba16_ep120.pt"
    model = KeypointDetector.load_from_checkpoint(model_file)
    return WrapperModel(model.adapter.model)


def export_as_onnx(model, input_sample):
    # datamodule = KeypointDetectionData.from_files(
    #     predict_files=predict_files,
    #     batch_size=4,
    # )
    # for d in datamodule.predict_dataset:
    #     print(d.keys())
    # trainer = Trainer()
    # predictions = trainer.predict(model, datamodule=datamodule)
    # input_sample = torch.randn((1, 64))
    input_names = ["input"]
    # output_names = ["output"]
    output_names = ["boxes", "labels", "scores", "keypoints"]  # , "rest_output"]

    # model("model.onnx", input_sample, export_params=True)

    # print(model)
    # for k in model.adapter.model.named_parameters():
    #     print(k)
    # print(model.adapter.model)
    model.eval()

    # myModel.qconfig = torch.quantization.default_qconfig
    # print(myModel.qconfig)
    # torch.quantization.prepare(myModel, inplace=True)

    # Convert to quantized model
    # torch.quantization.convert(myModel, inplace=True)

    torch.onnx.export(
        model,
        input_sample,
        "resnet18.onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
    )


def get_image_t(*, size):
    img = Image.open("data/samples/1.jpeg")
    img = img.resize((size, size))
    # transform = T.Compose([T.ToTensor(), T.Resize(size=(224, 224))])
    # t = transform(img)
    return torchvision.transforms.functional.to_tensor(img)


def main():
    # size = 224
    # input_sample = [torch.zeros((3, size, size))]
    input_sample = [get_image_t(size=224)]
    # print(input_sample.shape)
    export_as_onnx(make_wrapper_model(), input_sample)


main()
