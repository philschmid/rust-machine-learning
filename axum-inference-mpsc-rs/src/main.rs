use axum::{http::Method, routing::post, AddExtensionLayer, Router};
use std::net::SocketAddr;
use tokio::sync::mpsc;

use tower::ServiceBuilder;
use tower_http::{
    cors::{any, CorsLayer},
    trace::TraceLayer,
};

use crate::inference_engine::MpscInferencePayload;
use crate::predict_route::predict_route;

// custom modules
mod inference_engine;
mod predict_route;

// the input to our `create_user` handler

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // MPPSC Channel for sending Data to inference worker
    let (tx, rx) = mpsc::channel::<MpscInferencePayload>(32);

    // Set the RUST_LOG, if it hasn't been explicitly defined
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "axum_tokio_mpsc=debug,tower_http=debug")
    }
    // init request tracing for `TraceLayer`
    tracing_subscriber::fmt::init();

    // MPSC channel consumer for prediction
    tokio::spawn(async move { inference_engine::predict(rx).await });

    // middleware stack
    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(
            // see https://docs.rs/tower-http/latest/tower_http/cors/index.html
            // for more details
            CorsLayer::new()
                .allow_origin(any())
                .allow_methods(vec![Method::POST, Method::OPTIONS])
                .allow_headers(any()),
        )
        .layer(AddExtensionLayer::new(tx.clone()));

    // build our application with a route
    let app = Router::new()
        .route("/predict", post(predict_route))
        .layer(middleware_stack);

    // run it with hyper on localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .http1_preserve_header_case(true)
        .http1_title_case_headers(true)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}
