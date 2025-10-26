//! Streaming channels for LM Studio SDK compatibility.

use std::sync::Arc;

use dashmap::DashMap;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::protocol::messages::OutboundMessage;
use crate::protocol::types::KvConfigStack;
use crate::server::namespace::ApiNamespace;
use crate::server::websocket::ServerState;

mod llm;
mod stubs;
mod util;

#[derive(Clone, Hash, PartialEq, Eq)]
struct ChannelKey {
    connection_id: String,
    channel_id: i64,
}

#[derive(Clone)]
struct ChannelHandle {
    endpoint: String,
    cancel: CancellationToken,
    inbound_tx: mpsc::Sender<Value>,
}

pub struct ChannelControl {
    pub cancel: CancellationToken,
    pub inbound_rx: mpsc::Receiver<Value>,
}

/// Manages active streaming channels (connection-scoped IDs).
pub struct ChannelManager {
    channels: DashMap<ChannelKey, ChannelHandle>,
}

impl ChannelManager {
    pub fn new() -> Self {
        Self {
            channels: DashMap::new(),
        }
    }

    pub fn register(
        &self,
        connection_id: &str,
        channel_id: i64,
        endpoint: String,
    ) -> ChannelControl {
        let cancel = CancellationToken::new();
        let (inbound_tx, inbound_rx) = mpsc::channel::<Value>(32);
        self.channels.insert(
            ChannelKey {
                connection_id: connection_id.to_string(),
                channel_id,
            },
            ChannelHandle {
                endpoint,
                cancel: cancel.clone(),
                inbound_tx,
            },
        );
        ChannelControl { cancel, inbound_rx }
    }

    pub fn cancel(&self, connection_id: &str, channel_id: i64) -> Option<String> {
        let handle = self.channels.get(&ChannelKey {
            connection_id: connection_id.to_string(),
            channel_id,
        })?;
        handle.cancel.cancel();
        Some(handle.endpoint.clone())
    }

    pub fn forward_inbound(&self, connection_id: &str, channel_id: i64, message: Value) -> bool {
        let handle = self.channels.get(&ChannelKey {
            connection_id: connection_id.to_string(),
            channel_id,
        });
        let Some(handle) = handle else {
            return false;
        };
        handle.inbound_tx.try_send(message).is_ok()
    }

    pub fn remove(&self, connection_id: &str, channel_id: i64) -> Option<String> {
        let removed = self
            .channels
            .remove(&ChannelKey {
                connection_id: connection_id.to_string(),
                channel_id,
            })
            .map(|(_, c)| c)?;
        removed.cancel.cancel();
        Some(removed.endpoint)
    }

    pub fn remove_all_for_connection(&self, connection_id: &str) {
        let keys = self
            .channels
            .iter()
            .filter(|entry| entry.key().connection_id == connection_id)
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        for key in keys {
            if let Some((_, handle)) = self.channels.remove(&key) {
                handle.cancel.cancel();
            }
        }
    }
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Requests
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LoadModelChannelRequest {
    pub model_key: String,
    pub identifier: Option<String>,
    #[serde(rename = "ttlMs")]
    pub ttl_ms: Option<i64>,
    pub load_config_stack: KvConfigStack,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GetOrLoadChannelRequest {
    pub identifier: String,
    #[serde(rename = "loadTtlMs")]
    pub load_ttl_ms: Option<i64>,
    pub load_config_stack: KvConfigStack,
}

// ============================================================================
// Entry point
// ============================================================================

pub async fn handle_channel_create(
    namespace: ApiNamespace,
    connection_id: &str,
    channel_id: i64,
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
    tx: &mpsc::Sender<OutboundMessage>,
) {
    match namespace {
        ApiNamespace::Llm => {
            llm::handle_channel_create(connection_id, channel_id, endpoint, payload, state, tx)
                .await;
        }
        _ => {
            stubs::handle_channel_create(
                namespace,
                connection_id,
                channel_id,
                endpoint,
                payload,
                state,
                tx,
            )
            .await;
        }
    }
}
