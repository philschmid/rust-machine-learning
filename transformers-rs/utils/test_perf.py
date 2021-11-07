from transformers import AutoTokenizer

import torch
import time

input = "I love Rust and Pytorch"
model = torch.jit.load("model/model.pt")
tokenizer = AutoTokenizer.from_pretrained("model")
start = time.time()

tokens = tokenizer(
    input,
    return_tensors="pt",
)
sample_input = tuple(tokens.values())
model(*sample_input)

print(f"Prediction took {round(time.time() - start,6)* 1000}ms")
# Prediction took 155.65300000000002ms
