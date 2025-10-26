use serde::Deserialize;
use serde_json::json;

use crate::client::{ClientChannel, WsClient};
use crate::protocol::types::{ChatHistoryData, KvConfigStack, ModelSpecifier};

pub struct LlmClient<'a> {
    ws: &'a WsClient,
}

impl<'a> LlmClient<'a> {
    pub fn new(ws: &'a WsClient) -> Self {
        Self { ws }
    }

    pub async fn list_loaded(&self) -> anyhow::Result<Vec<LlmInstanceInfo>> {
        let result = self.ws.rpc_call("listLoaded", json!({})).await?;
        if result.is_null() {
            return Ok(Vec::new());
        }
        let models: Vec<LlmInstanceInfo> = serde_json::from_value(result)
            .map_err(|e| anyhow::anyhow!("Invalid listLoaded response: {}", e))?;
        Ok(models)
    }

    pub async fn unload_model(&self, identifier: &str) -> anyhow::Result<()> {
        let _ = self
            .ws
            .rpc_call("unloadModel", json!({ "identifier": identifier }))
            .await?;
        Ok(())
    }

    pub async fn load_model(
        &self,
        model_key: &str,
        load_config_stack: KvConfigStack,
    ) -> anyhow::Result<LoadModelStream> {
        let channel = self
            .ws
            .create_channel(
                "loadModel",
                json!({
                    "modelKey": model_key,
                    "loadConfigStack": load_config_stack,
                }),
            )
            .await?;
        Ok(LoadModelStream { channel })
    }

    pub async fn predict(
        &self,
        model_specifier: ModelSpecifier,
        history: ChatHistoryData,
        prediction_config_stack: KvConfigStack,
    ) -> anyhow::Result<PredictStream> {
        let channel = self
            .ws
            .create_channel(
                "predict",
                json!({
                    "modelSpecifier": model_specifier,
                    "history": history,
                    "predictionConfigStack": prediction_config_stack,
                }),
            )
            .await?;
        Ok(PredictStream { channel })
    }
}

pub struct LoadModelStream {
    channel: ClientChannel,
}

impl LoadModelStream {
    pub async fn recv(&mut self) -> Option<LoadModelEvent> {
        let msg = self.channel.recv().await?;
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match msg_type {
            "resolved" => Some(LoadModelEvent::Resolved),
            "progress" => {
                let progress = msg.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
                Some(LoadModelEvent::Progress { progress })
            }
            "success" => {
                let info_value = msg.get("info").cloned().unwrap_or_default();
                match serde_json::from_value(info_value) {
                    Ok(info) => Some(LoadModelEvent::Success { info }),
                    Err(_) => Some(LoadModelEvent::Unknown),
                }
            }
            _ => Some(LoadModelEvent::Unknown),
        }
    }
}

pub enum LoadModelEvent {
    Resolved,
    Progress { progress: f64 },
    Success { info: LlmInstanceInfo },
    Unknown,
}

pub struct PredictStream {
    channel: ClientChannel,
}

impl PredictStream {
    pub async fn recv(&mut self) -> Option<PredictEvent> {
        let msg = self.channel.recv().await?;
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match msg_type {
            "promptProcessingProgress" => Some(PredictEvent::PromptProcessingProgress),
            "fragment" => {
                let fragment_value = msg.get("fragment").cloned().unwrap_or_default();
                match serde_json::from_value(fragment_value) {
                    Ok(fragment) => Some(PredictEvent::Fragment(fragment)),
                    Err(_) => Some(PredictEvent::Unknown),
                }
            }
            "success" => Some(PredictEvent::Success),
            _ => Some(PredictEvent::Unknown),
        }
    }
}

pub enum PredictEvent {
    PromptProcessingProgress,
    Fragment(PredictFragment),
    Success,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictFragment {
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmInstanceInfo {
    pub identifier: String,
}
