//! Types derived from the vendored LM Studio SDK (`docs/lmstudio-sdk/`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// KVConfig
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvConfigField {
    pub key: String,
    #[serde(default)]
    pub value: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvConfig {
    pub fields: Vec<KvConfigField>,
}

impl KvConfig {
    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn with_field(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        let value = serde_json::to_value(value).unwrap_or(Value::Null);
        self.fields.push(KvConfigField {
            key: key.into(),
            value,
        });
        self
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvConfigStackLayer {
    pub layer_name: String,
    pub config: KvConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvConfigStack {
    pub layers: Vec<KvConfigStackLayer>,
}

// ============================================================================
// ModelSpecifier
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelQuery {
    pub domain: Option<String>,
    pub identifier: Option<String>,
    pub path: Option<String>,
    pub vision: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ModelSpecifier {
    Query {
        query: ModelQuery,
    },
    InstanceReference {
        #[serde(rename = "instanceReference")]
        instance_reference: String,
    },
}

// ============================================================================
// Chat History
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatHistoryData {
    pub messages: Vec<ChatMessageData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessageData {
    pub role: String,
    pub content: Vec<ChatMessagePartData>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatMessagePartData {
    Text {
        text: String,
    },
    File {
        name: String,
        identifier: String,
        #[serde(rename = "sizeBytes", default)]
        size_bytes: Option<i64>,
        #[serde(rename = "fileType", default)]
        file_type: Option<Value>,
    },
    ToolCallRequest {
        #[serde(rename = "toolCallRequest")]
        tool_call_request: Value,
    },
    ToolCallResult {
        content: String,
        #[serde(rename = "toolCallId", default)]
        tool_call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
    },
}

impl ChatMessagePartData {
    pub fn to_prompt_fragment(&self) -> &str {
        match self {
            Self::Text { text } => text,
            Self::File {
                name,
                identifier,
                size_bytes,
                file_type,
            } => {
                let _ = (name, identifier, size_bytes, file_type);
                ""
            }
            Self::ToolCallRequest { tool_call_request } => {
                let _ = tool_call_request;
                ""
            }
            Self::ToolCallResult {
                content,
                tool_call_id,
                name,
            } => {
                let _ = (content, tool_call_id, name);
                ""
            }
        }
    }
}

// ============================================================================
// LLM applyPromptTemplate opts
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmApplyPromptTemplateOpts {
    pub omit_bos_token: Option<bool>,
    pub omit_eos_token: Option<bool>,
    pub tool_definitions: Option<Vec<Value>>,
}
