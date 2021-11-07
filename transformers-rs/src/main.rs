use std::time::Instant;

use tch::{Device, Tensor};
use tokenizers::{
    tokenizer::{Result, Tokenizer},
    PaddingParams,
};

use crate::model::TorchscriptModel;

mod model;

trait ToTensor<T> {
    fn to_tensor(&self) -> Tensor
    where
        T: tch::kind::Element;
}

impl<T> ToTensor<T> for Vec<T> {
    fn to_tensor(&self) -> Tensor
    where
        T: tch::kind::Element,
    {
        Tensor::stack(&[Tensor::of_slice::<T>(self.as_slice())], 0)
    }
}

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_file("model/tokenizer.json")?
        .with_padding(Some(PaddingParams {
            // pad_to_multiple_of: Some(8u32 as usize),
            ..PaddingParams::default()
        }))
        .to_owned();

    let model = TorchscriptModel::new("model/model.pt");
    let max_len = 128;

    let input = "I love Rust and Pytorch";
    let now = Instant::now();

    let encoding = tokenizer.encode(input, true)?;

    // // TODO: check if output can be passed into model
    // encoding.pad(
    //     max_len,
    //     tokenizer.get_padding().unwrap().pad_id,
    //     tokenizer.get_padding().unwrap().pad_type_id,
    //     tokenizer.get_padding().unwrap().pad_token.as_str(),
    //     tokenizers::PaddingDirection::Right,
    // );

    let ids = encoding
        .get_ids()
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<i32>>()
        .to_tensor()
        .to_device(Device::Cpu);

    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<i32>>()
        .to_tensor()
        .to_device(Device::Cpu);

    // let token_type_ids = encoding
    //     .get_type_ids()
    //     .iter()
    //     .map(|x| *x as i32)
    //     .collect::<Vec<i32>>()
    //     .to_tensor();

    let predictions = model.forward(ids, attention_mask)?;
    println!(
        "Prediction took {} milliseconds.",
        now.elapsed().as_millis()
    );

    println!(
        "Class is {}",
        Vec::<u8>::from(predictions.argmax(1, false))[0]
    );

    Ok(())
}
