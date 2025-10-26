use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;

pub(super) async fn handle_rpc(
    endpoint: &str,
    payload: Value,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "reindexPlugins" => Ok(None),
        "processingHandleUpdate" => Ok(None),
        "processingHandleRequest" => {
            let req: ProcessingHandleRequestRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = (&req.pci, &req.token);
            let response = match req.request.get("type").and_then(Value::as_str) {
                Some("confirmToolCall") => json!({
                    "type": "confirmToolCall",
                    "result": { "type": "deny", "denyReason": "Plugin tool calls are not supported by this backend" }
                }),
                Some("textInput") => json!({ "type": "textInput", "result": "" }),
                _ => json!({ "type": "textInput", "result": "" }),
            };
            Ok(Some(json!({ "response": response })))
        }
        "processingPullHistory" => Ok(Some(json!({ "messages": [] }))),
        "processingGetOrLoadTokenSource" => Ok(Some(json!({
            "tokenSourceIdentifier": { "type": "model", "identifier": "unknown" }
        }))),
        "processingHasStatus" => Ok(Some(Value::Bool(false))),
        "processingNeedsNaming" => Ok(Some(Value::Bool(false))),
        "processingSuggestName" => Ok(None),
        "processingSetSenderName" => Ok(None),
        "setConfigSchematics" => Ok(None),
        "setGlobalConfigSchematics" => Ok(None),
        "pluginInitCompleted" => Ok(None),
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown plugins RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessingHandleRequestRequest {
    pub pci: String,
    pub token: String,
    pub request: Value,
}
