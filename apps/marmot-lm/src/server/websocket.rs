//! WebSocket server implementation

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::engine::model_manager::ModelManager;
use crate::protocol::messages::{InboundMessage, OutboundMessage};
use crate::server::channels::{handle_channel_create, ChannelManager};
use crate::server::namespace::ApiNamespace;
use crate::server::rpc::handle_rpc_call;
use crate::server::signals::SignalManager;

/// Server state shared across all connections
pub struct ServerState {
    pub model_manager: ModelManager,
    pub channel_manager: ChannelManager,
    pub signal_manager: SignalManager,
    pub shutdown: CancellationToken,
}

impl ServerState {
    fn new(shutdown: CancellationToken) -> Self {
        Self {
            model_manager: ModelManager::new(),
            channel_manager: ChannelManager::new(),
            signal_manager: SignalManager::new(),
            shutdown,
        }
    }
}

/// Discovery response for LM Studio SDK clients
#[derive(Serialize)]
struct GreetingResponse {
    lmstudio: bool,
}

/// Handle the discovery endpoint
async fn lmstudio_greeting() -> Json<GreetingResponse> {
    Json(GreetingResponse { lmstudio: true })
}

/// Handle WebSocket upgrade
async fn ws_handler_namespace(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
    Path(namespace): Path<String>,
) -> impl IntoResponse {
    let namespace = ApiNamespace::parse(&namespace);
    ws.on_upgrade(move |socket| handle_socket(socket, state, namespace))
}

async fn ws_handler_legacy(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state, ApiNamespace::Llm))
}

/// Handle a single WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<ServerState>, namespace: ApiNamespace) {
    let connection_id = Uuid::new_v4().to_string();
    tracing::info!(
        "New WebSocket connection: {} (namespace={})",
        connection_id,
        namespace.as_str()
    );

    let (mut sender, mut receiver) = socket.split();
    use futures_util::{SinkExt, StreamExt};

    // Authenticate (LM Studio SDK)
    let auth_ok = match receiver.next().await {
        Some(Ok(Message::Text(text))) => match serde_json::from_str::<AuthPacket>(&text) {
            Ok(packet) => {
                let _ = (&packet.client_identifier, &packet.client_passkey);
                if packet.auth_version != 1 {
                    let _ = sender
                        .send(Message::Text(
                            serde_json::json!({
                                "success": false,
                                "error": "Unsupported authVersion",
                            })
                            .to_string()
                            .into(),
                        ))
                        .await;
                    false
                } else {
                    let _ = sender
                        .send(Message::Text(
                            serde_json::json!({ "success": true }).to_string().into(),
                        ))
                        .await;
                    true
                }
            }
            Err(e) => {
                let _ = sender
                    .send(Message::Text(
                        serde_json::json!({
                            "success": false,
                            "error": format!("Failed to parse auth packet: {}", e),
                        })
                        .to_string()
                        .into(),
                    ))
                    .await;
                false
            }
        },
        Some(Ok(Message::Close(_))) | None => false,
        Some(Ok(_)) => {
            let _ = sender
                .send(Message::Text(
                    serde_json::json!({
                        "success": false,
                        "error": "Expected auth packet as first message",
                    })
                    .to_string()
                    .into(),
                ))
                .await;
            false
        }
        Some(Err(e)) => {
            tracing::warn!("WebSocket error during auth: {}", e);
            false
        }
    };

    if !auth_ok {
        tracing::info!(
            "Authentication failed/aborted: {} (namespace={})",
            connection_id,
            namespace.as_str()
        );
        return;
    }

    let (tx, mut rx) = mpsc::channel::<OutboundMessage>(100);

    // Spawn task to forward outbound messages
    let send_task = tokio::spawn(async move {
        use futures_util::SinkExt;
        while let Some(msg) = rx.recv().await {
            let json = match serde_json::to_string(&msg) {
                Ok(j) => j,
                Err(e) => {
                    tracing::error!("Failed to serialize message: {}", e);
                    continue;
                }
            };
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    let mut warning_count = 0u32;
    const MAX_WARNINGS: u32 = 5;

    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<InboundMessage>(&text) {
                    Ok(inbound) => {
                        handle_message(namespace, inbound, &connection_id, &state, &tx).await;
                    }
                    Err(e) => {
                        // Send communicationWarning for parse failures
                        if warning_count < MAX_WARNINGS {
                            warning_count += 1;
                            let warning = format!(
                                "Failed to parse message: {} (usually caused by protocol incompatibility)",
                                e
                            );
                            tracing::warn!("{}", warning);
                            let _ = tx
                                .send(OutboundMessage::CommunicationWarning { warning })
                                .await;

                            if warning_count == MAX_WARNINGS {
                                let _ = tx.send(OutboundMessage::CommunicationWarning {
                                    warning: format!("{} communication warnings have been produced. Further warnings will be suppressed.", MAX_WARNINGS),
                                }).await;
                            }
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                tracing::info!("Connection closed: {}", connection_id);
                break;
            }
            Ok(_) => {
                // Ignore binary/ping/pong
            }
            Err(e) => {
                tracing::error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    // Cleanup
    state.signal_manager.unsubscribe_all(&connection_id);
    state
        .channel_manager
        .remove_all_for_connection(&connection_id);
    send_task.abort();
    tracing::info!("Connection {} cleaned up", connection_id);
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuthPacket {
    #[serde(rename = "authVersion")]
    auth_version: i64,
    client_identifier: String,
    client_passkey: String,
}

/// Handle a parsed inbound message
async fn handle_message(
    namespace: ApiNamespace,
    msg: InboundMessage,
    connection_id: &str,
    state: &Arc<ServerState>,
    tx: &mpsc::Sender<OutboundMessage>,
) {
    match msg {
        InboundMessage::RpcCall {
            call_id,
            endpoint,
            parameter,
        } => match handle_rpc_call(namespace, &endpoint, parameter, state).await {
            Ok(result) => {
                let _ = tx
                    .send(OutboundMessage::RpcResult { call_id, result })
                    .await;
            }
            Err(error) => {
                let _ = tx.send(OutboundMessage::RpcError { call_id, error }).await;
            }
        },
        InboundMessage::ChannelCreate {
            channel_id,
            endpoint,
            creation_parameter,
        } => {
            tracing::info!("Channel create: {} -> {}", channel_id, endpoint);
            handle_channel_create(
                namespace,
                connection_id,
                channel_id,
                &endpoint,
                creation_parameter,
                state,
                tx,
            )
            .await;
        }
        InboundMessage::ChannelSend {
            channel_id,
            message,
            ack_id,
        } => {
            tracing::debug!("Channel send: {} {:?}", channel_id, message);
            // Send ack if requested
            if let Some(ack_id) = ack_id {
                let _ = tx
                    .send(OutboundMessage::ChannelAck { channel_id, ack_id })
                    .await;
            }

            let forwarded =
                state
                    .channel_manager
                    .forward_inbound(connection_id, channel_id, message.clone());
            if !forwarded {
                tracing::trace!("Channel send for unknown channel: {}", channel_id);
            }

            let msg_type = message.get("type").and_then(|t| t.as_str());
            let is_cancel_like = matches!(
                msg_type,
                Some(
                    "cancel" | "stop" | "cancelDownload" | "cancelPlan" | "discardSession" | "end"
                )
            );
            if is_cancel_like {
                if let Some(endpoint) = state.channel_manager.cancel(connection_id, channel_id) {
                    tracing::info!("Channel cancel: {} ({})", channel_id, endpoint);
                } else {
                    tracing::debug!("Channel cancel for unknown channel: {}", channel_id);
                }
            }
        }
        InboundMessage::ChannelAck { channel_id, ack_id } => {
            tracing::debug!("Channel ack: {} ack_id={}", channel_id, ack_id);
            // Acknowledgment received - could track pending messages here
        }
        InboundMessage::SignalSubscribe {
            subscribe_id,
            endpoint,
            creation_parameter,
        } => {
            tracing::debug!(
                "Signal subscribe: {} -> {} ({})",
                connection_id,
                endpoint,
                subscribe_id
            );
            state.signal_manager.subscribe(
                &endpoint,
                subscribe_id,
                connection_id,
                creation_parameter,
                tx.clone(),
            );
        }
        InboundMessage::SignalUnsubscribe { subscribe_id } => {
            tracing::debug!("Signal unsubscribe: {} -> {}", connection_id, subscribe_id);
            state
                .signal_manager
                .unsubscribe_by_id(connection_id, subscribe_id);
        }
        InboundMessage::WritableSignalSubscribe {
            subscribe_id,
            endpoint,
            creation_parameter,
        } => {
            tracing::debug!(
                "Writable signal subscribe: {} -> {} ({})",
                connection_id,
                endpoint,
                subscribe_id
            );
            state.signal_manager.subscribe_writable(
                &endpoint,
                subscribe_id,
                connection_id,
                creation_parameter,
                tx.clone(),
            );
        }
        InboundMessage::WritableSignalUnsubscribe { subscribe_id } => {
            tracing::debug!(
                "Writable signal unsubscribe: {} -> {}",
                connection_id,
                subscribe_id
            );
            state
                .signal_manager
                .unsubscribe_writable_by_id(connection_id, subscribe_id);
        }
        InboundMessage::WritableSignalUpdate {
            subscribe_id,
            patches,
            tags,
        } => {
            tracing::debug!(
                "Writable signal update: {} -> {} ({} tags)",
                connection_id,
                subscribe_id,
                tags.len()
            );
            state
                .signal_manager
                .apply_writable_update(connection_id, subscribe_id, patches, tags);
        }
        InboundMessage::KeepAlive => {
            tracing::trace!("KeepAlive from {}", connection_id);
            let _ = tx.send(OutboundMessage::KeepAliveAck).await;
        }
        InboundMessage::CommunicationWarning { warning } => {
            tracing::warn!("Communication warning from {}: {}", connection_id, warning);
        }
    }
}

/// Start the WebSocket server with shutdown support
pub async fn start_server_with_shutdown(
    port: u16,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let state = Arc::new(ServerState::new(shutdown.clone()));

    // Spawn TTL auto-unload background task
    let ttl_state = Arc::clone(&state);
    let ttl_shutdown = shutdown.clone();
    tokio::spawn(async move {
        run_ttl_checker(ttl_state, ttl_shutdown).await;
    });

    let app = Router::new()
        .route("/lmstudio-greeting", get(lmstudio_greeting))
        .route("/ws", get(ws_handler_legacy))
        .route("/{namespace}", get(ws_handler_namespace))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting server on {}", addr);
    tracing::info!(
        "Discovery endpoint: http://localhost:{}/lmstudio-greeting",
        port
    );
    tracing::info!("WebSocket endpoints:");
    tracing::info!("  - ws://localhost:{}/llm", port);
    tracing::info!("  - ws://localhost:{}/system", port);
    tracing::info!("  - ws://localhost:{}/embedding", port);
    tracing::info!("  - ws://localhost:{}/diagnostics", port);
    tracing::info!("  - ws://localhost:{}/files", port);
    tracing::info!("  - ws://localhost:{}/repository", port);
    tracing::info!("  - ws://localhost:{}/plugins", port);
    tracing::info!("  - ws://localhost:{}/runtime", port);
    tracing::info!(
        "WebSocket alias: ws://localhost:{}/ws (alias for /llm)",
        port
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown.cancelled().await;
            tracing::info!("Received shutdown signal, stopping server...");
        })
        .await?;

    tracing::info!("Server stopped");
    Ok(())
}

/// Background task to check and unload models past their TTL
async fn run_ttl_checker(state: Arc<ServerState>, shutdown: CancellationToken) {
    let check_interval = std::time::Duration::from_secs(60); // Check every minute

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                tracing::debug!("TTL checker shutting down");
                break;
            }
            _ = tokio::time::sleep(check_interval) => {
                let unloaded = state.model_manager.check_ttl_unload();
                if !unloaded.is_empty() {
                    // Update loaded models signal
                    let loaded_ids = state.model_manager.list_loaded()
                        .iter()
                        .map(|m| m.identifier.clone())
                        .collect();
                    state.signal_manager.update_loaded_models(loaded_ids);

                    for id in unloaded {
                        tracing::info!("Auto-unloaded model '{}' due to TTL expiration", id);
                    }
                }
            }
        }
    }
}
