import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image


def test1(model_filename):
    onnx_model = onnx.load(model_filename)
    onnx.checker.check_model(onnx_model)


def test2(model_filename):
    ort_sess = ort.InferenceSession(model_filename)
    # size = 224
    # input = np.zeros((3, size, size), dtype=np.float32)
    img = Image.open("data/samples/1.jpeg")
    transform = T.Compose([T.ToTensor(), T.Resize(size=(224, 224))])
    input = transform(img)

    outputs = ort_sess.run(None, {"input": input.detach().numpy()})
    # Print Result
    boxes, labels, scores, keypoints, _rest = outputs
    # for t in outputs:
    #     print(t.shape)
    print(keypoints[0])
    # print(f"This: output={outputs}")


if __name__ == "__main__":
    model_filename = "resnet18.onnx"
    test1(model_filename)
    test2(model_filename)
