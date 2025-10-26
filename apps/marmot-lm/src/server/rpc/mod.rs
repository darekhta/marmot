//! RPC handler for LM Studio SDK compatibility.
//!
//! Source of truth: `docs/lmstudio-sdk/packages/lms-external-backend-interfaces/src/*BackendInterface.ts`

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;
use crate::server::namespace::ApiNamespace;
use crate::server::websocket::ServerState;

mod files;
mod format;
mod llm;
mod plugins;
mod repository;
mod runtime;
mod system;
mod types;

pub use format::format_messages_with_defaults;

pub async fn handle_rpc_call(
    namespace: ApiNamespace,
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
) -> Result<Option<Value>, ProtocolError> {
    tracing::debug!(
        "RPC call: namespace={} endpoint={} payload={}",
        namespace.as_str(),
        endpoint,
        payload
    );

    match namespace {
        ApiNamespace::System => system::handle_rpc(endpoint, payload, state).await,
        ApiNamespace::Llm => llm::handle_rpc(endpoint, payload, state).await,
        ApiNamespace::Embedding => Err(ProtocolError::with_suggestion(
            "Namespace not implemented",
            "Embedding namespace is not yet implemented",
            "Use the LLM namespace instead",
        )),
        ApiNamespace::Diagnostics => handle_diagnostics_rpc(endpoint).await,
        ApiNamespace::Files => files::handle_rpc(endpoint, payload).await,
        ApiNamespace::Repository => repository::handle_rpc(endpoint, payload).await,
        ApiNamespace::Plugins => plugins::handle_rpc(endpoint, payload).await,
        ApiNamespace::Runtime => runtime::handle_rpc(endpoint, payload).await,
        _ => Err(ProtocolError::with_suggestion(
            "Unsupported namespace",
            format!("Namespace '{}' is not implemented", namespace.as_str()),
            "Use a supported API namespace",
        )),
    }
}

async fn handle_diagnostics_rpc(endpoint: &str) -> Result<Option<Value>, ProtocolError> {
    Err(ProtocolError::with_suggestion(
        "Endpoint not found",
        format!("Unknown diagnostics RPC endpoint: {}", endpoint),
        "Diagnostics namespace only supports channel endpoints",
    ))
}

fn parse_payload<T: for<'de> Deserialize<'de>>(payload: Value) -> Result<T, serde_json::Error> {
    serde_json::from_value(payload)
}

fn invalid_request(err: serde_json::Error) -> ProtocolError {
    ProtocolError::with_cause("Invalid request", err.to_string())
}

fn base64_size_bytes(input: &str) -> i64 {
    let trimmed = input.trim();
    let len = trimmed.len() as i64;
    let without_pad = trimmed.trim_end_matches('=').len() as i64;
    let pad = len.saturating_sub(without_pad);
    ((len * 3) / 4).saturating_sub(pad)
}

fn downloaded_model_to_llm_info(model: crate::engine::model_manager::DownloadedModelInfo) -> Value {
    json!({
        "type": "llm",
        "modelKey": model.path,
        "format": "gguf",
        "displayName": model.name,
        "publisher": "local",
        "path": model.path,
        "sizeBytes": model.size_bytes,
        "quantization": crate::common::quantization_to_value(model.quantization.as_deref()),
        "vision": false,
        "trainedForToolUse": false,
        "maxContextLength": 2048,
    })
}

fn loaded_model_to_llm_instance_info(info: &crate::engine::model_manager::FullModelInfo) -> Value {
    json!({
        "type": "llm",
        "modelKey": info.path,
        "format": info.format,
        "displayName": info.display_name,
        "publisher": "local",
        "path": info.path,
        "sizeBytes": info.size_bytes,
        "paramsString": info.params_string,
        "architecture": info.architecture,
        "quantization": crate::common::quantization_to_value(info.quantization.as_deref()),
        "identifier": info.identifier,
        "instanceReference": format!("instance:{}", info.identifier),
        "ttlMs": Value::Null,
        "lastUsedTime": crate::common::unix_time_ms(),
        "vision": info.vision,
        "trainedForToolUse": info.trained_for_tool_use,
        "maxContextLength": info.max_context_length,
        "contextLength": info.context_length,
    })
}

fn estimate_model_usage(model_key: &str) -> Value {
    let size_bytes = std::fs::metadata(model_key).map(|m| m.len()).unwrap_or(0) as f64;
    json!({
        "memory": {
            "confidence": "low",
            "modelVramBytes": 0.0,
            "contextVramBytes": 0.0,
            "totalVramBytes": 0.0,
            "modelBytes": size_bytes,
            "contextBytes": 0.0,
            "totalBytes": size_bytes,
        },
        "passesGuardrails": true,
    })
}

fn guess_file_type(name: &str) -> Value {
    let extension = std::path::Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    match extension.to_ascii_lowercase().as_str() {
        "md" => json!({ "type": "markdown" }),
        "txt" => json!({ "type": "plainText" }),
        "pdf" => json!({ "type": "pdf" }),
        _ => json!({ "type": "unknown" }),
    }
}

const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
