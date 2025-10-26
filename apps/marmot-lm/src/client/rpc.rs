use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};

use super::transport::WsTransport;

#[derive(Clone)]
pub struct RpcManager {
    state: Arc<RpcState>,
}

struct RpcState {
    pending: Mutex<HashMap<i64, oneshot::Sender<Value>>>,
    next_id: AtomicI64,
}

impl RpcManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RpcState {
                pending: Mutex::new(HashMap::new()),
                next_id: AtomicI64::new(0),
            }),
        }
    }

    pub async fn call(
        &self,
        transport: &WsTransport,
        endpoint: &str,
        parameter: Value,
    ) -> anyhow::Result<Value> {
        let call_id = self.state.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "type": "rpcCall",
            "callId": call_id,
            "endpoint": endpoint,
            "parameter": parameter,
        });

        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.state.pending.lock().await;
            pending.insert(call_id, tx);
        }

        transport.send(&msg).await?;
        let response = rx.await?;

        if response.get("type").and_then(|t| t.as_str()) == Some("rpcError") {
            if let Some(error) = response.get("error") {
                let title = error
                    .get("title")
                    .and_then(|t| t.as_str())
                    .unwrap_or("Unknown");
                let message = error.get("cause").and_then(|m| m.as_str()).unwrap_or("");
                anyhow::bail!("RPC error: {} - {}", title, message);
            }
        }

        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn handle_message(&self, msg: &Value) {
        if let Some(call_id) = msg.get("callId").and_then(|v| v.as_i64()) {
            let sender = {
                let mut pending = self.state.pending.lock().await;
                pending.remove(&call_id)
            };
            if let Some(sender) = sender {
                let _ = sender.send(msg.clone());
            }
        }
    }
}
