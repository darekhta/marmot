use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use super::transport::WsTransport;

#[derive(Clone)]
pub struct SignalManager {
    state: Arc<SignalState>,
}

struct SignalState {
    subscriptions: Mutex<HashMap<i64, mpsc::Sender<Value>>>,
    writable_subscriptions: Mutex<HashMap<i64, mpsc::Sender<Value>>>,
    next_subscribe_id: AtomicI64,
    next_writable_subscribe_id: AtomicI64,
}

pub struct SignalSubscription {
    id: i64,
    rx: mpsc::Receiver<Value>,
}

impl SignalSubscription {
    pub async fn recv(&mut self) -> Option<Value> {
        self.rx.recv().await
    }

    pub fn id(&self) -> i64 {
        self.id
    }
}

pub struct WritableSignalSubscription {
    id: i64,
    rx: mpsc::Receiver<Value>,
}

impl WritableSignalSubscription {
    pub async fn recv(&mut self) -> Option<Value> {
        self.rx.recv().await
    }

    pub fn id(&self) -> i64 {
        self.id
    }
}

impl SignalManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(SignalState {
                subscriptions: Mutex::new(HashMap::new()),
                writable_subscriptions: Mutex::new(HashMap::new()),
                next_subscribe_id: AtomicI64::new(0),
                next_writable_subscribe_id: AtomicI64::new(0),
            }),
        }
    }

    pub async fn subscribe(
        &self,
        transport: &WsTransport,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<SignalSubscription> {
        let subscribe_id = self.state.next_subscribe_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "type": "signalSubscribe",
            "endpoint": endpoint,
            "subscribeId": subscribe_id,
            "creationParameter": creation_parameter,
        });

        let (tx, rx) = mpsc::channel(32);
        {
            let mut subs = self.state.subscriptions.lock().await;
            subs.insert(subscribe_id, tx);
        }

        transport.send(&msg).await?;
        Ok(SignalSubscription {
            id: subscribe_id,
            rx,
        })
    }

    pub async fn unsubscribe(
        &self,
        transport: &WsTransport,
        subscribe_id: i64,
    ) -> anyhow::Result<()> {
        let msg = json!({
            "type": "signalUnsubscribe",
            "subscribeId": subscribe_id,
        });

        {
            let mut subs = self.state.subscriptions.lock().await;
            subs.remove(&subscribe_id);
        }

        transport.send(&msg).await?;
        Ok(())
    }

    pub async fn subscribe_writable(
        &self,
        transport: &WsTransport,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<WritableSignalSubscription> {
        let subscribe_id = self
            .state
            .next_writable_subscribe_id
            .fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "type": "writableSignalSubscribe",
            "endpoint": endpoint,
            "subscribeId": subscribe_id,
            "creationParameter": creation_parameter,
        });

        let (tx, rx) = mpsc::channel(32);
        {
            let mut subs = self.state.writable_subscriptions.lock().await;
            subs.insert(subscribe_id, tx);
        }

        transport.send(&msg).await?;
        Ok(WritableSignalSubscription {
            id: subscribe_id,
            rx,
        })
    }

    pub async fn unsubscribe_writable(
        &self,
        transport: &WsTransport,
        subscribe_id: i64,
    ) -> anyhow::Result<()> {
        let msg = json!({
            "type": "writableSignalUnsubscribe",
            "subscribeId": subscribe_id,
        });

        {
            let mut subs = self.state.writable_subscriptions.lock().await;
            subs.remove(&subscribe_id);
        }

        transport.send(&msg).await?;
        Ok(())
    }

    pub async fn update_writable(
        &self,
        transport: &WsTransport,
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
        let msg = json!({
            "type": "writableSignalUpdate",
            "subscribeId": subscribe_id,
            "patches": patches,
            "tags": tags,
        });

        transport.send(&msg).await?;
        Ok(())
    }

    pub async fn handle_update(&self, msg: &Value) {
        let Some(subscribe_id) = msg.get("subscribeId").and_then(|v| v.as_i64()) else {
            return;
        };
        let sender = {
            let subscriptions = self.state.subscriptions.lock().await;
            subscriptions.get(&subscribe_id).cloned()
        };

        if let Some(sender) = sender {
            let _ = sender.send(msg.clone()).await;
        }
    }

    pub async fn handle_error(&self, msg: &Value) {
        let Some(subscribe_id) = msg.get("subscribeId").and_then(|v| v.as_i64()) else {
            return;
        };

        if let Some(error) = msg.get("error") {
            let title = error
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown");
            let message = error.get("cause").and_then(|m| m.as_str()).unwrap_or("");
            tracing::error!("Signal {} error: {} - {}", subscribe_id, title, message);
        }

        let mut subscriptions = self.state.subscriptions.lock().await;
        subscriptions.remove(&subscribe_id);
    }

    pub async fn handle_writable_update(&self, msg: &Value) {
        let Some(subscribe_id) = msg.get("subscribeId").and_then(|v| v.as_i64()) else {
            return;
        };
        let sender = {
            let subscriptions = self.state.writable_subscriptions.lock().await;
            subscriptions.get(&subscribe_id).cloned()
        };

        if let Some(sender) = sender {
            let _ = sender.send(msg.clone()).await;
        }
    }

    pub async fn handle_writable_error(&self, msg: &Value) {
        let Some(subscribe_id) = msg.get("subscribeId").and_then(|v| v.as_i64()) else {
            return;
        };

        if let Some(error) = msg.get("error") {
            let title = error
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown");
            let message = error.get("cause").and_then(|m| m.as_str()).unwrap_or("");
            tracing::error!(
                "Writable signal {} error: {} - {}",
                subscribe_id,
                title,
                message
            );
        }

        let mut subscriptions = self.state.writable_subscriptions.lock().await;
        subscriptions.remove(&subscribe_id);
    }
}
