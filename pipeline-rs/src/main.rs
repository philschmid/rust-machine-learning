use std::path::Path;

use anyhow::Result;
use lib::tasks::{text_classification::TextClassificationPipeline, Pipeline};
// use tokenizers::Tokenizer;

fn benchmark(seq_len: u8, mut pipeline: TextClassificationPipeline) {
    let loops = 1000;
    let mut seq = String::new();
    for _ in 0..seq_len {
        seq.push_str(" a");
    }
    let start = std::time::Instant::now();
    for _ in 0..loops {
        pipeline.call(&seq).unwrap();
    }
    let end = start.elapsed();
    println!(
        "Benchmark: seq_len={}; avg={}µs",
        seq_len,
        end.as_micros() / loops
    );
}

fn main() -> Result<()> {
    let path = Path::new("model");
    // let model = OnnxModel::from_path(path)?;
    // let tokenizer = Tokenizer::from_path(path)?;
    // let mut pipeline = TextClassificationPipeline::new(model, tokenizer);
    let mut pipeline = TextClassificationPipeline::from_path(path)?;

    let input = "I like you. I love you.";
    // let input = "I hate you.";
    let start_time = std::time::Instant::now();
    let res = pipeline.call(input)?;
    println!("Inference took {}ms", start_time.elapsed().as_millis());
    println!("{:?}", res);
    benchmark(64, pipeline);
    Ok(())
}
