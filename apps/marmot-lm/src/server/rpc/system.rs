use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;
use crate::server::websocket::ServerState;

pub(super) async fn handle_rpc(
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "listDownloadedModels" => {
            let models = state
                .model_manager
                .list_downloaded()
                .into_iter()
                .map(super::downloaded_model_to_llm_info)
                .collect::<Vec<_>>();
            Ok(Some(Value::Array(models)))
        }
        "listDownloadedModelVariants" => {
            let req: ListDownloadedModelVariantsRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = &req.model_key;
            Ok(Some(Value::Array(Vec::new())))
        }
        "alive" => Ok(None),
        "notify" => {
            let _notification = payload;
            Ok(None)
        }
        "version" => Ok(Some(
            json!({ "version": super::SERVER_VERSION, "build": 1 }),
        )),
        "setExperimentFlag" => {
            let req: SetExperimentFlagRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = (&req.code, req.value);
            Ok(None)
        }
        "getExperimentFlags" => Ok(Some(Value::Array(Vec::new()))),
        "startHttpServer" => {
            let req: StartHttpServerRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = (req.port, req.cors, req.network_interface.as_deref());
            Ok(None)
        }
        "stopHttpServer" => Ok(None),
        "info" => Ok(Some(json!({
            "pid": std::process::id(),
            "isDaemon": false,
            "version": super::SERVER_VERSION,
        }))),
        "requestShutdown" => {
            state.shutdown.cancel();
            Ok(None)
        }
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown system RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListDownloadedModelVariantsRequest {
    pub model_key: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetExperimentFlagRequest {
    pub code: String,
    pub value: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartHttpServerRequest {
    pub port: Option<i64>,
    pub cors: Option<bool>,
    pub network_interface: Option<String>,
}
