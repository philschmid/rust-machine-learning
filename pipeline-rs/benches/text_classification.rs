use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lib::tasks::{text_classification::TextClassificationPipeline, Pipeline};

fn text_classification_benchmark(c: &mut Criterion) {
    let mut pipeline = TextClassificationPipeline::from_path(Path::new("model")).unwrap();
    let sequence_length = vec![8, 16, 32, 64, 128, 256, 512];
    // sequence length 8

    for seq in &sequence_length {
        c.bench_function(
            format!("preprocessing;sequence_length {}", seq).as_str(),
            |b| b.iter(|| pipeline.preprocess(black_box::<&str>(" I".repeat(seq - 2).as_str()))),
        );
        let processed_input = pipeline.preprocess(" I".repeat(seq - 2).as_str()).unwrap();

        c.bench_function(
            format!("prediction;sequence_length {}", seq).as_str(),
            |b| b.iter(|| pipeline.predict(black_box(processed_input.clone()))),
        );
        let predicted_output = pipeline
            .predict(pipeline.preprocess(" I".repeat(seq - 2).as_str()).unwrap())
            .unwrap();
        c.bench_function(
            format!("postprocessing;sequence_length {}", seq).as_str(),
            |b| b.iter(|| pipeline.postprocess(black_box(predicted_output.clone()))),
        );
        c.bench_function(format!("e2e;sequence_length {}", seq).as_str(), |b| {
            b.iter(|| pipeline.call(black_box::<&str>(" I".repeat(seq - 2).as_str())))
        });
    }
}

criterion_group!(benches, text_classification_benchmark);
criterion_main!(benches);
