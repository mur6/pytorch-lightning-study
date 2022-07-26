# 4. Detect objects in a few images!
import pickle
import sys
from pathlib import Path

from flash import Trainer
from flash.image import KeypointDetectionData, KeypointDetector


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
    return predictions


if __name__ == "__main__":
    model_file = sys.argv[1]
    print(f"model_file={model_file}")
    model = KeypointDetector.load_from_checkpoint(model_file)
    files = get_predict_files(Path(sys.argv[2]))
    # print(files)
    pred = main(model, files)
    with open("predictions.pkl", "wb") as f:
        pickle.dump(pred, f)
