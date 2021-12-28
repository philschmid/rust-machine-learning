use std::{thread::sleep, time::Duration};
use tokenizers::{tokenizer::Tokenizer, PaddingParams, Result};
use tokio::sync::{mpsc, oneshot};

use crate::predict_route::PredictRequest;

use tch::{no_grad, Device, Tensor};

pub struct MpscInferencePayload {
    pub payload: PredictRequest,
    pub resp: oneshot::Sender<String>,
}

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

pub async fn predict(mut rx: mpsc::Receiver<MpscInferencePayload>) {
    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .unwrap()
        .with_padding(Some(PaddingParams {
            // pad_to_multiple_of: Some(8u32 as usize),
            ..PaddingParams::default()
        }))
        .to_owned();

    let model = TorchscriptModel::new("model/model.pt");

    while let Some(payload) = rx.recv().await {
        // TODO: is misisng copy trait
        // tokio::task::spawn_blocking(move || {
        let encoding = tokenizer.encode(payload.payload.inputs, true).unwrap();

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

        let predictions = model.forward(ids, attention_mask).unwrap();
        let label = Vec::<u8>::from(predictions.argmax(1, false))[0];
        let _ = payload.resp.send(label.to_string());
        model
        // })
        // .await
        // .unwrap();
    }
}

pub struct TorchscriptModel {
    model: tch::CModule,
    device: Device,
}

impl TorchscriptModel {
    pub fn new(model_path: &str) -> Self {
        let model = tch::CModule::load(model_path).unwrap();
        let device = Device::cuda_if_available();

        TorchscriptModel { model, device }
    }
    pub fn forward(&self, ids: Tensor, attention_mask: Tensor) -> Result<Tensor> {
        let ids = tch::IValue::from(ids);
        let attention_mask = tch::IValue::from(attention_mask);

        let output = no_grad(|| self.model.forward_is(&[ids, attention_mask]))?;

        match output {
            tch::IValue::Tuple(t) => match &t[0] {
                tch::IValue::Tensor(t) => Ok(t.detach().to_device(self.device)),
                _ => panic!("Expected a tensor"),
            },
            _ => panic!("Expected a tensor"),
        }
    }
}
