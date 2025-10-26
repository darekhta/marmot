use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::protocol::messages::{OutboundMessage, ProtocolError};
use crate::server::channels::ChannelControl;
use crate::server::namespace::ApiNamespace;
use crate::server::websocket::ServerState;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateArtifactDownloadPlanChannelRequest {
    pub owner: String,
    pub name: String,
}

pub(super) async fn handle_channel_create(
    namespace: ApiNamespace,
    connection_id: &str,
    channel_id: i64,
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
    tx: &mpsc::Sender<OutboundMessage>,
) {
    match namespace {
        ApiNamespace::System => match endpoint {
            "alive" => {
                let _ =
                    state
                        .channel_manager
                        .register(connection_id, channel_id, endpoint.to_string());
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown system channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        ApiNamespace::Diagnostics => match endpoint {
            "streamLogs" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_idle_channel(connection_id, channel_id, state, tx, cancel).await;
                });
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown diagnostics channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        ApiNamespace::Files => match endpoint {
            "retrieve" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_files_retrieve_channel(connection_id, channel_id, state, tx, cancel).await;
                });
            }
            "parseDocument" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_files_parse_document_channel(connection_id, channel_id, state, tx, cancel)
                        .await;
                });
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown files channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        ApiNamespace::Repository => match endpoint {
            "downloadModel" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_repository_download_model_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "downloadArtifact" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_repository_download_artifact_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "pushArtifact" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_repository_push_artifact_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "ensureAuthenticated" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_repository_ensure_authenticated_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "createArtifactDownloadPlan" => {
                let req: CreateArtifactDownloadPlanChannelRequest =
                    match serde_json::from_value(payload) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx
                                .send(OutboundMessage::ChannelError {
                                    channel_id,
                                    error: ProtocolError::with_cause(
                                        "Invalid request",
                                        format!(
                                            "Invalid createArtifactDownloadPlan payload: {}",
                                            e
                                        ),
                                    ),
                                })
                                .await;
                            return;
                        }
                    };

                let ChannelControl { cancel, inbound_rx } =
                    state
                        .channel_manager
                        .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_repository_create_artifact_download_plan_channel(
                        connection_id,
                        channel_id,
                        req,
                        state,
                        tx,
                        cancel,
                        inbound_rx,
                    )
                    .await;
                });
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown repository channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        ApiNamespace::Plugins => match endpoint {
            "startToolUseSession" => {
                let ChannelControl { cancel, inbound_rx } =
                    state
                        .channel_manager
                        .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_plugins_start_tool_use_session_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                        inbound_rx,
                    )
                    .await;
                });
            }
            "generateWithGenerator" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_plugins_generate_with_generator_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "registerDevelopmentPlugin" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_plugins_register_development_plugin_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            "setPromptPreprocessor"
            | "setPredictionLoopHandler"
            | "setToolsProvider"
            | "setGenerator" => {
                let _ =
                    state
                        .channel_manager
                        .register(connection_id, channel_id, endpoint.to_string());
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown plugins channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        ApiNamespace::Runtime => match endpoint {
            "downloadRuntimeExtension" => {
                let ChannelControl {
                    cancel,
                    inbound_rx: _inbound_rx,
                } = state
                    .channel_manager
                    .register(connection_id, channel_id, endpoint.to_string());

                let connection_id = connection_id.to_string();
                let state = Arc::clone(state);
                let tx = tx.clone();
                tokio::spawn(async move {
                    run_runtime_download_runtime_extension_channel(
                        connection_id,
                        channel_id,
                        state,
                        tx,
                        cancel,
                    )
                    .await;
                });
            }
            _ => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause(
                            "Unknown endpoint",
                            format!("Unknown runtime channel endpoint: {}", endpoint),
                        ),
                    })
                    .await;
            }
        },
        _ => {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause(
                        "Unsupported namespace",
                        format!("Namespace '{}' is not supported", namespace.as_str()),
                    ),
                })
                .await;
        }
    }
}

async fn run_idle_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    cancel.cancelled().await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    let _ = state
        .channel_manager
        .remove(connection_id.as_str(), channel_id);
}

async fn run_files_retrieve_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({
                "type": "result",
                "result": { "entries": [] }
            })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_files_parse_document_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let parser = json!({ "library": "marmot-lm", "version": env!("CARGO_PKG_VERSION") });
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "parserLoaded", "parser": parser })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "progress", "progress": 1.0 })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({
                "type": "result",
                "content": "",
                "parser": { "library": "marmot-lm", "version": env!("CARGO_PKG_VERSION") }
            })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_repository_download_model_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let update = json!({ "downloadedBytes": 0, "totalBytes": 0, "speedBytesPerSecond": 0.0 });
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "downloadProgress", "update": update })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "startFinalizing" })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "success", "defaultIdentifier": "default" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_repository_download_artifact_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let update = json!({ "downloadedBytes": 0, "totalBytes": 0, "speedBytesPerSecond": 0.0 });
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "downloadProgress", "update": update })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "startFinalizing" })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "success" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_repository_push_artifact_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "message", "message": "pushArtifact is not supported" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_repository_ensure_authenticated_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "authenticated" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_repository_create_artifact_download_plan_channel(
    connection_id: String,
    channel_id: i64,
    req: CreateArtifactDownloadPlanChannelRequest,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
    mut inbound_rx: mpsc::Receiver<Value>,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let plan = json!({
        "nodes": [
            {
                "type": "artifact",
                "owner": req.owner,
                "name": req.name,
                "state": "pending",
                "dependencyNodes": [],
            }
        ],
        "downloadSizeBytes": 0,
    });

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "planReady", "plan": plan })),
            ack_id: None,
        })
        .await;

    let mut committed = false;
    loop {
        tokio::select! {
            _ = cancel.cancelled() => { break; }
            msg = inbound_rx.recv() => {
                let Some(msg) = msg else { break; };
                if msg.get("type").and_then(Value::as_str) == Some("commit") {
                    committed = true;
                    break;
                }
            }
        }
    }

    if committed && !cancel.is_cancelled() {
        let update = json!({ "downloadedBytes": 0, "totalBytes": 0, "speedBytesPerSecond": 0.0 });
        let _ = tx
            .send(OutboundMessage::ChannelSend {
                channel_id,
                message: Some(json!({ "type": "downloadProgress", "update": update })),
                ack_id: None,
            })
            .await;
        let _ = tx
            .send(OutboundMessage::ChannelSend {
                channel_id,
                message: Some(json!({ "type": "startFinalizing" })),
                ack_id: None,
            })
            .await;
        let _ = tx
            .send(OutboundMessage::ChannelSend {
                channel_id,
                message: Some(json!({ "type": "success" })),
                ack_id: None,
            })
            .await;
    }

    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_plugins_start_tool_use_session_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
    mut inbound_rx: mpsc::Receiver<Value>,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "sessionReady", "toolDefinitions": [] })),
            ack_id: None,
        })
        .await;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => { break; }
            msg = inbound_rx.recv() => {
                let Some(msg) = msg else { break; };
                match msg.get("type").and_then(Value::as_str) {
                    Some("callTool") => {
                        let call_id = msg.get("callId").and_then(Value::as_i64).unwrap_or(0);
                        let name = msg.get("name").and_then(Value::as_str).unwrap_or("unknown");
                        let error = ProtocolError::with_cause(
                            "Tool call not supported",
                            format!("Tool '{}' is not available", name),
                        );
                        let error_value = serde_json::to_value(error).unwrap_or(Value::Null);
                        let _ = tx.send(OutboundMessage::ChannelSend{
                            channel_id,
                            message: Some(json!({
                                "type": "toolCallError",
                                "callId": call_id,
                                "error": error_value
                            })),
                            ack_id: None,
                        }).await;
                    }
                    Some("discardSession") => { break; }
                    _ => {}
                }
            }
        }
    }

    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_plugins_generate_with_generator_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "success" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_plugins_register_development_plugin_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    let client_identifier = Uuid::new_v4().to_string();
    let client_passkey = Uuid::new_v4().to_string();
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({
                "type": "ready",
                "clientIdentifier": client_identifier,
                "clientPasskey": client_passkey,
            })),
            ack_id: None,
        })
        .await;

    cancel.cancelled().await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}

async fn run_runtime_download_runtime_extension_channel(
    connection_id: String,
    channel_id: i64,
    state: Arc<ServerState>,
    tx: mpsc::Sender<OutboundMessage>,
    cancel: CancellationToken,
) {
    let connection_id_ref = connection_id.as_str();
    let mut done = false;
    let mut cleanup = || {
        if !done {
            let _ = state.channel_manager.remove(connection_id_ref, channel_id);
            done = true;
        }
    };

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let update = json!({ "downloadedBytes": 0, "totalBytes": 0, "speedBytesPerSecond": 0.0 });
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "downloadProgress", "update": update })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "startFinalizing" })),
            ack_id: None,
        })
        .await;
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "success" })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
    cleanup();
}
