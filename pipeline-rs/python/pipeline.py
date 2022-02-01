from transformers import AutoTokenizer
import time
import onnxruntime as ort
import numpy as np
import timeit

sess_options = ort.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# classifier = pipeline("text-classification", device=0)
ort_session = ort.InferenceSession("model/model.onnx", sess_options, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("model/")


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def process(inputs):
    return tokenizer(inputs, return_tensors="np")


def predict(onnx_inputs):
    return ort_session.run(None, onnx_inputs.data)[0]


def postprocess(onnx_outputs):
    scores = softmax(onnx_outputs)
    return {"label": scores.argmax().item(), "score": scores.max().item()}


def clx(inputs):
    onnx_inputs = process(inputs)
    onnx_outputs = predict(onnx_inputs)
    return postprocess(onnx_outputs)


if __name__ == "__main__":
    seq_lenghts = [8, 16, 32, 64, 128, 256, 512]
    for seq_len in seq_lenghts:
        loops = 1000
        b_i = " l" * (seq_len - 2)
        b_p = process(b_i)
        b_o = predict(b_p)
        duration_preprocess = (
            (timeit.timeit(f"process('{b_i}')", globals=locals(), number=loops) / loops) * 1000 * 1000
        )
        duration_predict = (timeit.timeit(f"predict(b_p)", globals=locals(), number=loops) / loops) * 1000 * 1000
        duration_postprocess = (
            (timeit.timeit(f"postprocess(b_o)", globals=locals(), number=loops) / loops) * 1000 * 1000
        )
        duration_e2e = (timeit.timeit(f"clx('{b_i}')", globals=locals(), number=loops) / loops) * 1000 * 1000

        print(f"############# Start of benchmark ###############")
        print(f"Benchmark for sequence length: {seq_len}:")
        print(f"Avg preprocess time: {duration_preprocess}µs")
        print(f"Avg predict time: {duration_predict}µs")
        print(f"Avg postprocess time: {duration_postprocess}µs")
        print(f"-----------------------------------------------")
        print(f"Avg e2e time: {duration_e2e}µs")
        print(f"############# End of benchmark ###############")

    # benchmark(8)
    # benchmark(16)
    # benchmark(32)
    # benchmark(64)
    # benchmark(128)
    # benchmark(256)
