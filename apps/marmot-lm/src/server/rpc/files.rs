use serde::Deserialize;
use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;

pub(super) async fn handle_rpc(
    endpoint: &str,
    payload: Value,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "getLocalFileAbsolutePath" => {
            let req: GetLocalFileAbsolutePathRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            Ok(Some(json!({ "path": req.file_name })))
        }
        "uploadFileBase64" => {
            let req: UploadFileBase64Request =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            Ok(Some(json!({
                "identifier": format!("upload:{}", uuid::Uuid::new_v4()),
                "fileType": super::guess_file_type(&req.name),
                "sizeBytes": super::base64_size_bytes(&req.content_base64),
            })))
        }
        "getDocumentParsingLibrary" => {
            let req: GetDocumentParsingLibraryRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = &req.file_identifier;
            Ok(Some(
                json!({ "library": "marmot-lm", "version": super::SERVER_VERSION }),
            ))
        }
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown files RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetLocalFileAbsolutePathRequest {
    pub file_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadFileBase64Request {
    pub name: String,
    pub content_base64: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetDocumentParsingLibraryRequest {
    pub file_identifier: String,
}
