from transformers import AutoTokenizer
import time
import onnxruntime as ort
import numpy as np

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


def benchmark(seq_len, pipeline):
    loops = 1000
    start = time.time()
    for i in range(loops):
        pipeline(" l" * seq_len)
    end = time.time()

    print(f"Benchmark: seq_len={seq_len}; avg={((end-start)/loops)*1000*1000}µs")


def clx(inputs):
    onnx_inputs = tokenizer(inputs, return_tensors="np")
    onnx_outputs = ort_session.run(None, onnx_inputs.data)[0]
    scores = softmax(onnx_outputs)
    res = {"label": scores.argmax().item(), "score": scores.max().item()}
    return res


if __name__ == "__main__":
    benchmark(8, clx)
    benchmark(16, clx)
    benchmark(32, clx)
    benchmark(64, clx)
    benchmark(128, clx)
    benchmark(256, clx)
