use serde::Deserialize;
use serde_json::json;

use crate::client::WsClient;

pub struct SystemClient<'a> {
    ws: &'a WsClient,
}

impl<'a> SystemClient<'a> {
    pub fn new(ws: &'a WsClient) -> Self {
        Self { ws }
    }

    pub async fn version(&self) -> anyhow::Result<SystemVersion> {
        let result = self.ws.rpc_call("version", json!({})).await?;
        serde_json::from_value(result)
            .map_err(|e| anyhow::anyhow!("Invalid system version response: {}", e))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemVersion {
    pub version: String,
    pub build: i64,
}
