# 4. Detect objects in a few images!
import argparse
import pickle
from pathlib import Path

import torch
from flash import Trainer
from flash.image import KeypointDetectionData, KeypointDetector


def export(model):
    # datamodule = KeypointDetectionData.from_files(
    #     predict_files=predict_files,
    #     batch_size=4,
    # )
    # for d in datamodule.predict_dataset:
    #     print(d.keys())
    # trainer = Trainer()
    # predictions = trainer.predict(model, datamodule=datamodule)
    input_sample = torch.randn((1, 64))
    input_names = ["input"]
    output_names = ["output"]

    # model("model.onnx", input_sample, export_params=True)
    input_sample = torch.randn(1, 3, 128, 128)
    input_sample = [torch.rand(3, 224, 224)]  # , torch.rand(3, 500, 400)]
    # print(model)
    # for k in model.adapter.model.named_parameters():
    #     print(k)
    # print(model.adapter.model)
    myModel = model.adapter.model
    myModel.eval()

    # myModel.qconfig = torch.quantization.default_qconfig
    # print(myModel.qconfig)
    # torch.quantization.prepare(myModel, inplace=True)

    # Convert to quantized model
    # torch.quantization.convert(myModel, inplace=True)

    torch.onnx.export(
        myModel,
        input_sample,
        "resnet18.onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
    )


model_file = "models/kypt_resnet18_fpn_ba16_ep120.pt"
model = KeypointDetector.load_from_checkpoint(model_file)
export(model)
