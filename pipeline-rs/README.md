# Pipeline modules implemented with `ONNX` and `tokenizers`


```bash
cargo build --release

cargo run --release

```

### Benchmark

To run benchmarks is used [Citerion](https://github.com/bheisler/criterion.rs) with `cargo bench`. 

Run

```bash
cargo bench
```

#### Results 

`Cascade-lake` cpu
```bash
model name      : Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz
physical id     : 0
siblings        : 4
core id         : 0
cpu cores       : 2
apicid          : 0
initial apicid  : 0
```

without defining `with_number_threads`:

```bash
Benchmark: seq_len=8; avg=3335µs
Benchmark: seq_len=16; avg=4542µs
Benchmark: seq_len=32; avg=7166µs
Benchmark: seq_len=64; avg=12930µs
Benchmark: seq_len=128; avg=22862µs
```

python tests
```bash
Benchmark: seq_len=8; avg=3194.565534591675µs
Benchmark: seq_len=16; avg=4181.376695632935µs
Benchmark: seq_len=32; avg=6296.816349029541µs
Benchmark: seq_len=64; avg=10360.423564910889µs
Benchmark: seq_len=128; avg=19073.987007141113µs
Benchmark: seq_len=256; avg=39876.47366523743µs
```

## Resources

* [softmax](https://github.com/CasperN/drug/blob/1a7cc4532aa4bdb7ce091a53d2d6b14ab2d5a0dd/src/lib.rs#L77)
* [argmax: statistical methods for ndarray's ArrayBase type.](https://github.com/rust-ndarray/ndarray-stats)
* [ndarray](https://github.com/rust-ndarray/ndarray)

## Todo

* [ ] add `cargo bench`
* [x] wrap `tokenizer` for easier error handle and configuration for GPU
* [x] add `from_path` to `Pipeline` trait
* [ ] add batching
* [ ] add configuration parameters, like all scores, or only the top-k
* [ ] test Ice-lake and other models