use anyhow::{anyhow, bail, Result};
use tch::{no_grad, Device, Tensor};
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
                _ => bail!("Expected a tensor"),
            },
            _ => bail!("Expected a tensor"),
        }
    }
}
