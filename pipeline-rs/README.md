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

```bash
text_classification;sequence_length 8                                                                             
                        time:   [4.2265 ms 4.2444 ms 4.2624 ms]
                        change: [-1.8932% -0.0775% +1.1158%] (p = 0.95 > 0.05)
                        No change in performance detected.

text_classification;sequence_length 16                                                                             
                        time:   [3.9316 ms 3.9689 ms 4.0118 ms]
                        change: [-31.492% -26.790% -22.297%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) high mild
  5 (5.00%) high severe

text_classification;sequence_length 32                                                                            
                        time:   [9.5959 ms 9.6266 ms 9.6600 ms]
Found 4 outliers among 100 measurements (4.00%)
  3 (3.00%) high mild
  1 (1.00%) high severe

text_classification;sequence_length 64                                                                            
                        time:   [9.9642 ms 10.048 ms 10.155 ms]
Found 6 outliers among 100 measurements (6.00%)
  1 (1.00%) high mild
  5 (5.00%) high severe

text_classification;sequence_length 128                                                                            
                        time:   [18.948 ms 19.131 ms 19.361 ms]
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) high mild
  6 (6.00%) high severe

text_classification;sequence_length 256                                                                            
                        time:   [39.450 ms 39.971 ms 40.642 ms]
Found 11 outliers among 100 measurements (11.00%)
  3 (3.00%) high mild
  8 (8.00%) high severe

Benchmarking text_classification;sequence_length 512
text_classification;sequence_length 512                                                                            
                        time:   [92.450 ms 93.526 ms 94.926 ms]
Found 10 outliers among 100 measurements (10.00%)
  2 (2.00%) high mild
  8 (8.00%) high severe
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