use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;

pub(super) async fn handle_rpc(
    endpoint: &str,
    _payload: Value,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "searchModels" => Ok(Some(json!({ "results": [] }))),
        "getModelDownloadOptions" => Ok(Some(json!({ "results": [] }))),
        "installPluginDependencies" => Ok(None),
        "getLocalArtifactFiles" => Ok(Some(json!({
            "fileList": { "files": [], "usedIgnoreFile": Value::Null }
        }))),
        "loginWithPreAuthenticatedKeys" => Ok(Some(json!({ "userName": "local" }))),
        "installLocalPlugin" => Ok(None),
        "getModelCatalog" => Ok(Some(json!({ "models": [] }))),
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown repository RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}
