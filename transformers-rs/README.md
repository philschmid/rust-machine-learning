# Hugging Face Transformers using (Py)Torch amd Rust

## Examples

- [saving Python model & run inference with rust](./examples/python-to-rust)

## Resources

- [torch rust bindings](https://github.com/LaurentMazare/tch-rs)
- [Loading and Running a PyTorch Model in Rust](https://github.com/LaurentMazare/tch-rs/tree/master/examples/jit)

## Getting started

Get libtorch from the [PyTorch website download section](https://pytorch.org/get-started/locally/) and extract the content of the zip file.

```bash
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

cargo run
```

## compile and run

build

```Bash
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
cargo build --release
```

run

```bash
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
./target/release/transformers-rs
```

## Comparison with Python

```bash
input = "I love Rust and Pytorch"
model = "distilbert-base-uncased-finetuned-sst-2-english"
```

| input                     | Python | Rust Debug | Rust Release |
| ------------------------- | ------ | ---------- | ------------ |
| `I love Rust and Pytorch` | 155ms  | 170ms      | 152ms        |
|                           |        |            |              |
