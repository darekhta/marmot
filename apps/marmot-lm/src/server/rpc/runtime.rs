use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;

pub(super) async fn handle_rpc(
    endpoint: &str,
    _payload: Value,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "listEngines" => Ok(Some(Value::Array(Vec::new()))),
        "getEngineSelections" => Ok(Some(json!({
            "json": [],
            "meta": { "values": ["map"] }
        }))),
        "selectEngine" => Ok(None),
        "removeEngine" => Ok(None),
        "surveyHardware" => Ok(Some(
            json!({ "status": "NoCompatibleBackends", "engines": [] }),
        )),
        "searchRuntimeExtensions" => Ok(Some(json!({ "extensions": [] }))),
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown runtime RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}
