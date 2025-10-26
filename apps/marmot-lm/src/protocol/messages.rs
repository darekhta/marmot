//! Protocol message types for LM Studio SDK compatibility
//!
//! Source of truth: `docs/lmstudio-sdk/packages/lms-communication/src/Transport.ts`

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Inbound messages from client to server
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum InboundMessage {
    // Communication
    #[serde(rename = "communicationWarning")]
    CommunicationWarning { warning: String },
    #[serde(rename = "keepAlive")]
    KeepAlive,

    // Channel
    #[serde(rename = "channelCreate")]
    ChannelCreate {
        endpoint: String,
        #[serde(rename = "channelId")]
        channel_id: i64,
        #[serde(rename = "creationParameter", default)]
        creation_parameter: Value,
    },
    #[serde(rename = "channelSend")]
    ChannelSend {
        #[serde(rename = "channelId")]
        channel_id: i64,
        #[serde(default)]
        message: Value,
        #[serde(rename = "ackId")]
        ack_id: Option<i64>,
    },
    #[serde(rename = "channelAck")]
    ChannelAck {
        #[serde(rename = "channelId")]
        channel_id: i64,
        #[serde(rename = "ackId")]
        ack_id: i64,
    },

    // RPC
    #[serde(rename = "rpcCall")]
    RpcCall {
        endpoint: String,
        #[serde(rename = "callId")]
        call_id: i64,
        #[serde(default)]
        parameter: Value,
    },

    // Read-only signal
    #[serde(rename = "signalSubscribe")]
    SignalSubscribe {
        #[serde(rename = "creationParameter", default)]
        creation_parameter: Value,
        endpoint: String,
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
    },
    #[serde(rename = "signalUnsubscribe")]
    SignalUnsubscribe {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
    },

    // Writable signal
    #[serde(rename = "writableSignalSubscribe")]
    WritableSignalSubscribe {
        #[serde(rename = "creationParameter", default)]
        creation_parameter: Value,
        endpoint: String,
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
    },
    #[serde(rename = "writableSignalUnsubscribe")]
    WritableSignalUnsubscribe {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
    },
    #[serde(rename = "writableSignalUpdate")]
    WritableSignalUpdate {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    },
}

/// Outbound messages from server to client
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutboundMessage {
    // Communication
    #[serde(rename = "communicationWarning")]
    CommunicationWarning { warning: String },
    #[serde(rename = "keepAliveAck")]
    KeepAliveAck,

    // Channel
    #[serde(rename = "channelSend")]
    ChannelSend {
        #[serde(rename = "channelId")]
        channel_id: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<Value>,
        #[serde(rename = "ackId", skip_serializing_if = "Option::is_none")]
        ack_id: Option<i64>,
    },
    #[serde(rename = "channelAck")]
    ChannelAck {
        #[serde(rename = "channelId")]
        channel_id: i64,
        #[serde(rename = "ackId")]
        ack_id: i64,
    },
    #[serde(rename = "channelClose")]
    ChannelClose {
        #[serde(rename = "channelId")]
        channel_id: i64,
    },
    #[serde(rename = "channelError")]
    ChannelError {
        #[serde(rename = "channelId")]
        channel_id: i64,
        error: ProtocolError,
    },

    // RPC
    #[serde(rename = "rpcResult")]
    RpcResult {
        #[serde(rename = "callId")]
        call_id: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
    },
    #[serde(rename = "rpcError")]
    RpcError {
        #[serde(rename = "callId")]
        call_id: i64,
        error: ProtocolError,
    },

    // Read-only signal
    #[serde(rename = "signalUpdate")]
    SignalUpdate {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    },
    #[serde(rename = "signalError")]
    SignalError {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
        error: ProtocolError,
    },

    // Writable signal
    #[serde(rename = "writableSignalUpdate")]
    WritableSignalUpdate {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    },
    #[serde(rename = "writableSignalError")]
    WritableSignalError {
        #[serde(rename = "subscribeId")]
        subscribe_id: i64,
        error: ProtocolError,
    },
}

/// Protocol error structure matching `serializedLMSExtendedErrorSchema`
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProtocolError {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_title: Option<String>,
}

impl ProtocolError {
    pub fn new(title: impl Into<String>) -> Self {
        let title = title.into();
        Self {
            title: title.clone(),
            cause: None,
            suggestion: None,
            error_data: None,
            display_data: None,
            stack: None,
            root_title: Some(title),
        }
    }

    pub fn with_cause(title: impl Into<String>, cause: impl Into<String>) -> Self {
        let title = title.into();
        Self {
            title: title.clone(),
            cause: Some(cause.into()),
            suggestion: None,
            error_data: None,
            display_data: None,
            stack: None,
            root_title: Some(title),
        }
    }

    pub fn with_suggestion(
        title: impl Into<String>,
        cause: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        let title = title.into();
        Self {
            title: title.clone(),
            cause: Some(cause.into()),
            suggestion: Some(suggestion.into()),
            error_data: None,
            display_data: None,
            stack: None,
            root_title: Some(title),
        }
    }
}
