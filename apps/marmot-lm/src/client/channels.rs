use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use super::transport::WsTransport;

#[derive(Clone)]
pub struct ChannelManager {
    state: Arc<ChannelState>,
}

struct ChannelState {
    active: Mutex<HashMap<i64, mpsc::Sender<Value>>>,
    next_id: AtomicI64,
}

pub struct ClientChannel {
    rx: mpsc::Receiver<Value>,
}

impl ClientChannel {
    pub async fn recv(&mut self) -> Option<Value> {
        self.rx.recv().await
    }
}

impl ChannelManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(ChannelState {
                active: Mutex::new(HashMap::new()),
                next_id: AtomicI64::new(0),
            }),
        }
    }

    pub async fn create(
        &self,
        transport: &WsTransport,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<ClientChannel> {
        let channel_id = self.state.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "type": "channelCreate",
            "channelId": channel_id,
            "endpoint": endpoint,
            "creationParameter": creation_parameter,
        });

        let (tx, rx) = mpsc::channel(32);
        {
            let mut channels = self.state.active.lock().await;
            channels.insert(channel_id, tx);
        }

        transport.send(&msg).await?;
        Ok(ClientChannel { rx })
    }

    pub async fn handle_send(&self, msg: &Value) {
        let Some(channel_id) = msg.get("channelId").and_then(|v| v.as_i64()) else {
            return;
        };
        let sender = {
            let channels = self.state.active.lock().await;
            channels.get(&channel_id).cloned()
        };

        if let Some(sender) = sender {
            if let Some(message) = msg.get("message") {
                let _ = sender.send(message.clone()).await;
            }
        }
    }

    pub async fn handle_close(&self, msg: &Value) {
        let Some(channel_id) = msg.get("channelId").and_then(|v| v.as_i64()) else {
            return;
        };
        let mut channels = self.state.active.lock().await;
        channels.remove(&channel_id);
    }

    pub async fn handle_error(&self, msg: &Value) {
        let Some(channel_id) = msg.get("channelId").and_then(|v| v.as_i64()) else {
            return;
        };

        if let Some(error) = msg.get("error") {
            let title = error
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown");
            let message = error.get("cause").and_then(|m| m.as_str()).unwrap_or("");
            tracing::error!("Channel {} error: {} - {}", channel_id, title, message);
        }

        let mut channels = self.state.active.lock().await;
        channels.remove(&channel_id);
    }
}
