use serde::Deserialize;

use crate::protocol::types::{
    ChatHistoryData, KvConfigStack, LlmApplyPromptTemplateOpts, ModelSpecifier,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct UnloadModelRequest {
    pub identifier: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GetModelInfoRequest {
    pub specifier: ModelSpecifier,
    pub throw_if_not_found: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GetLoadConfigRequest {
    pub specifier: ModelSpecifier,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GetBasePredictionConfigRequest {
    pub specifier: ModelSpecifier,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GetInstanceProcessingStateRequest {
    pub specifier: ModelSpecifier,
    pub throw_if_not_found: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct EstimateModelUsageRequest {
    pub model_key: String,
    pub load_config_stack: KvConfigStack,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ApplyPromptTemplateRequest {
    pub specifier: ModelSpecifier,
    pub history: ChatHistoryData,
    pub prediction_config_stack: KvConfigStack,
    pub opts: LlmApplyPromptTemplateOpts,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TokenizeRequest {
    pub specifier: ModelSpecifier,
    pub input_string: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CountTokensRequest {
    pub specifier: ModelSpecifier,
    pub input_string: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PreloadDraftModelRequest {
    pub specifier: ModelSpecifier,
    pub draft_model_key: String,
}
