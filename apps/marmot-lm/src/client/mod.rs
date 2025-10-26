//! WebSocket client for connecting to marmot-lm server

use serde_json::{json, Value};

mod channels;
mod rpc;
mod signals;
mod transport;

pub mod api;

pub use api::{LlmClient, SystemClient};
pub use channels::ClientChannel;
pub use signals::{SignalSubscription, WritableSignalSubscription};

use channels::ChannelManager;
use rpc::RpcManager;
use signals::SignalManager;
use transport::WsTransport;

/// WebSocket client for CLI→server communication
pub struct WsClient {
    transport: WsTransport,
    rpc: RpcManager,
    channels: ChannelManager,
    signals: SignalManager,
}

impl WsClient {
    /// Try to connect to a marmot-lm server
    pub async fn connect(port: u16, namespace: &str) -> anyhow::Result<Self> {
        let (transport, mut incoming) = WsTransport::connect(port, namespace).await?;

        let rpc = RpcManager::new();
        let channels = ChannelManager::new();
        let signals = SignalManager::new();

        let rpc_handle = rpc.clone();
        let channels_handle = channels.clone();
        let signals_handle = signals.clone();

        tokio::spawn(async move {
            while let Some(msg) = incoming.recv().await {
                let msg_type = msg.get("type").and_then(|v| v.as_str());
                match msg_type {
                    Some("rpcResult") | Some("rpcError") => {
                        rpc_handle.handle_message(&msg).await;
                    }
                    Some("channelSend") => {
                        channels_handle.handle_send(&msg).await;
                    }
                    Some("channelClose") => {
                        channels_handle.handle_close(&msg).await;
                    }
                    Some("channelError") => {
                        channels_handle.handle_error(&msg).await;
                    }
                    Some("signalUpdate") => {
                        signals_handle.handle_update(&msg).await;
                    }
                    Some("signalError") => {
                        signals_handle.handle_error(&msg).await;
                    }
                    Some("writableSignalUpdate") => {
                        signals_handle.handle_writable_update(&msg).await;
                    }
                    Some("writableSignalError") => {
                        signals_handle.handle_writable_error(&msg).await;
                    }
                    Some("communicationWarning") => {
                        if let Some(warning) = msg.get("warning").and_then(|w| w.as_str()) {
                            tracing::warn!("Communication warning: {}", warning);
                        }
                    }
                    Some("keepAliveAck") => {
                        tracing::trace!("KeepAlive acknowledged");
                    }
                    Some("channelAck") => {}
                    Some(other) => {
                        tracing::debug!("Unknown message type: {}", other);
                    }
                    None => {
                        tracing::debug!("Message without type field");
                    }
                }
            }
        });

        Ok(Self {
            transport,
            rpc,
            channels,
            signals,
        })
    }

    /// Make an RPC call to the server
    pub async fn rpc_call(&self, endpoint: &str, parameter: Value) -> anyhow::Result<Value> {
        self.rpc.call(&self.transport, endpoint, parameter).await
    }

    /// Create a streaming channel
    pub async fn create_channel(
        &self,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<ClientChannel> {
        self.channels
            .create(&self.transport, endpoint, creation_parameter)
            .await
    }

    pub async fn subscribe_signal(&self, endpoint: &str) -> anyhow::Result<SignalSubscription> {
        self.subscribe_signal_with_parameter(endpoint, Value::Null)
            .await
    }

    pub async fn subscribe_signal_with_parameter(
        &self,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<SignalSubscription> {
        self.signals
            .subscribe(&self.transport, endpoint, creation_parameter)
            .await
    }

    pub async fn unsubscribe_signal(&self, subscribe_id: i64) -> anyhow::Result<()> {
        self.signals
            .unsubscribe(&self.transport, subscribe_id)
            .await
    }

    pub async fn subscribe_writable_signal(
        &self,
        endpoint: &str,
    ) -> anyhow::Result<WritableSignalSubscription> {
        self.subscribe_writable_signal_with_parameter(endpoint, Value::Null)
            .await
    }

    pub async fn subscribe_writable_signal_with_parameter(
        &self,
        endpoint: &str,
        creation_parameter: Value,
    ) -> anyhow::Result<WritableSignalSubscription> {
        self.signals
            .subscribe_writable(&self.transport, endpoint, creation_parameter)
            .await
    }

    pub async fn unsubscribe_writable_signal(&self, subscribe_id: i64) -> anyhow::Result<()> {
        self.signals
            .unsubscribe_writable(&self.transport, subscribe_id)
            .await
    }

    pub async fn update_writable_signal(
        &self,
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    ) -> anyhow::Result<()> {
        self.signals
            .update_writable(&self.transport, subscribe_id, patches, tags)
            .await
    }

    pub async fn send_keepalive(&self) -> anyhow::Result<()> {
        let msg = json!({ "type": "keepAlive" });
        self.transport.send(&msg).await
    }

    pub async fn send_communication_warning(&self, warning: &str) -> anyhow::Result<()> {
        let msg = json!({ "type": "communicationWarning", "warning": warning });
        self.transport.send(&msg).await
    }

    pub fn llm(&self) -> LlmClient<'_> {
        LlmClient::new(self)
    }

    pub fn system(&self) -> SystemClient<'_> {
        SystemClient::new(self)
    }

    /// Close the client connection
    pub async fn close(&self) -> anyhow::Result<()> {
        self.transport.close().await
    }
}

/// Try to connect to server, returns None if server is not available
pub async fn try_connect_server(port: u16, namespace: &str) -> Option<WsClient> {
    WsClient::connect(port, namespace).await.ok()
}
