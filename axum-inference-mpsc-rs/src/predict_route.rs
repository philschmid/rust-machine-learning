use axum::{extract::Extension, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::inference_engine::MpscInferencePayload;

#[derive(Serialize, Debug)]
struct PredictResponse {
    label: String,
    score: f32,
}

#[derive(Deserialize, Debug)]
pub struct PredictRequest {
    pub inputs: String,
}

pub async fn predict_route(
    Json(payload): Json<PredictRequest>,
    Extension(tx_clone): Extension<mpsc::Sender<MpscInferencePayload>>,
) -> impl IntoResponse {
    // create oneshot channel for recieving response from compute_heavy
    let (resp_tx, resp_rx) = oneshot::channel::<String>();

    // send data here
    tracing::debug!("Got {}", payload.inputs);
    let _ = tx_clone
        .send(MpscInferencePayload {
            payload,
            resp: resp_tx,
        })
        .await;
    // Await the response
    let res = resp_rx.await;
    println!("{:?}", res);
    Json(PredictResponse {
        label: "1".to_string(),
        score: 0.2,
    })
}
