import os
import torch
from transformers import pipeline

model = "distilbert-base-uncased-finetuned-sst-2-english"
task = "text-classification"
seq_len = 128
nlp = pipeline(task=task, model=model, tokenizer=model, framework="pt", model_kwargs={"torchscript": True})
assert nlp.model.config.torchscript is True
tokens = nlp.tokenizer(
    "This is a sample output", return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True
)
sample_input = tuple(tokens.values())

traced_script_module = torch.jit.trace(nlp.model, sample_input)
traced_script_module.save(os.path.join("model", "model.pt"))
nlp.tokenizer.save_pretrained("model")
