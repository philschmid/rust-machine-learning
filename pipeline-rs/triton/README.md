# Triton BLS Example of Text-Classification pipeline with Hugging Face x ORT


## Build Docker

```Bash
docker build -t triton-bls-example .
```

## Start Triton

```bash
	docker run  -t -i	-p 8000:8000 \
  -v $(pwd)/models:/opt/tritonserver/models \
  -v $(pwd)/tokenizer.json:/tmp/transformers/tokenizer.json \
  triton-bls-example \
  tritonserver --model-repository=/opt/tritonserver/models
```

## Run client



## Resources

* [BLS example documentation](https://github.com/triton-inference-server/python_backend/tree/main/examples/bls)
* [blog](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c)