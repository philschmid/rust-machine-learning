use std::path::Path;

use anyhow::{bail, Result};
use lib::{
    base::Pipeline, modeling_utils::OnnxModel,
    tasks::text_classification::TextClassificationPipeline,
};
use tokenizers::Tokenizer;

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
    let model = OnnxModel::from_file(path)?;
    let tokenizer = match Tokenizer::from_file(path.join("tokenizer.json")) {
        Ok(tk) => tk,
        Err(err) => bail!("{}", err),
    };

    let mut pipeline = TextClassificationPipeline::new(model, tokenizer);

    let input = "I like you. I love you.";
    // let input = "I hate you.";
    let start_time = std::time::Instant::now();
    let res = pipeline.call(input)?;
    println!("Inference took {}ms", start_time.elapsed().as_millis());
    println!("{:?}", res);
    benchmark(64, pipeline);
    Ok(())
}
