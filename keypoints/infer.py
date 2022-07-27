# 4. Detect objects in a few images!
import argparse
import pickle
from pathlib import Path

from flash import Trainer
from flash.image import KeypointDetectionData, KeypointDetector


def get_predict_files(sample_image_dir):
    iter = sample_image_dir.glob("*.jpeg")
    iter = map(str, sorted(iter))
    # print(iter)
    return list(iter)


def infer(model, predict_files):
    datamodule = KeypointDetectionData.from_files(
        predict_files=predict_files,
        batch_size=4,
    )
    trainer = Trainer()
    predictions = trainer.predict(model, datamodule=datamodule)
    return predictions


def main(*, model_file, sample_image_dir, output_file):
    print(f"model_file={model_file} output_file={output_file}")
    model = KeypointDetector.load_from_checkpoint(model_file)
    predict_files = get_predict_files(sample_image_dir)
    pred = infer(model, predict_files)
    output_file.write_bytes(pickle.dumps(pred))


parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=Path, required=True)
parser.add_argument("--sample_image_dir", type=Path, required=True)
parser.add_argument("--output_pkl_file", type=Path, default="data/outputs/predictions.pkl")

if __name__ == "__main__":
    args = parser.parse_args()
    main(model_file=args.model_file, sample_image_dir=args.sample_image_dir, output_file=args.output_pkl_file)
