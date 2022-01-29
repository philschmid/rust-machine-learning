use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lib::tasks::{text_classification::TextClassificationPipeline, Pipeline};

fn text_classification_benchmark(c: &mut Criterion) {
    let mut pipeline = TextClassificationPipeline::from_path(Path::new("model")).unwrap();
    // sequence length 8
    c.bench_function("text_classification;sequence_length 8", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(8).as_str())))
    });
    // sequence length 16
    c.bench_function("text_classification;sequence_length 16", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(16).as_str())))
    });
    // sequence length 32
    c.bench_function("text_classification;sequence_length 32", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(32).as_str())))
    });
    // sequence length 64
    c.bench_function("text_classification;sequence_length 64", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(64).as_str())))
    });
    // sequence length 128
    c.bench_function("text_classification;sequence_length 128", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(128).as_str())))
    });
    // sequence length 256
    c.bench_function("text_classification;sequence_length 256", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(256).as_str())))
    });
    // sequence length 512
    c.bench_function("text_classification;sequence_length 512", |b| {
        b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(512).as_str())))
    });
}

criterion_group!(benches, text_classification_benchmark);
criterion_main!(benches);
