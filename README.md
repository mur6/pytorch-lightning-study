# pytorch-lightning-study
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -r requirements/main.txt
(.venv) $ pip install -r requirements/dev.txt
```

## 学習
```
python keypoints/train.py --data_dir data/coco_keypoints/ --batch_size 4 --max_epochs 10
```

## 推論
```
python keypoints/infer.py --model_file models/keypoint_ba16_ep250.pt --sample_image_dir data/samples/
```
