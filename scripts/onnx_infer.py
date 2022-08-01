import numpy as np
import onnx
import onnxruntime as ort


def test1(model_filename):
    onnx_model = onnx.load(model_filename)
    onnx.checker.check_model(onnx_model)


def test2(model_filename):
    ort_sess = ort.InferenceSession(model_filename)
    # input = np.random.random((1, 3, 28, 28), dtype=np.float32)
    size = 224
    outputs = ort_sess.run(None, {"input": np.zeros((3, size, size), dtype=np.float32)})
    # Print Result
    for t in outputs:
        print(t)
    # print(f"This: output={outputs}")


if __name__ == "__main__":
    model_filename = "resnet18.onnx"
    test1(model_filename)
    test2(model_filename)
