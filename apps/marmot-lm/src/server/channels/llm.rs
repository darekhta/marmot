use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::collapse_kv_stack;
use crate::engine::model_manager::{GpuOffload, LoadConfig};
use crate::ffi::{GenerateOptions, RequestState, SamplingOptions, SamplingPreset};
use crate::protocol::messages::{OutboundMessage, ProtocolError};
use crate::protocol::types::{ChatHistoryData, KvConfigStack, ModelSpecifier};
use crate::server::channels::util::{
    apply_load_config_stack, default_identifier_from_key, kv_config_from_stack,
    llm_info_for_model_key, llm_instance_info_from_full,
};
use crate::server::channels::{ChannelControl, GetOrLoadChannelRequest, LoadModelChannelRequest};
use crate::server::model_utils::resolve_loaded_model;
use crate::server::namespace::ApiNamespace;
use crate::server::websocket::ServerState;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PredictChannelRequest {
    pub model_specifier: ModelSpecifier,
    pub history: ChatHistoryData,
    pub prediction_config_stack: KvConfigStack,
    pub fuzzy_preset_identifier: Option<String>,
    pub ignore_server_session_config: Option<bool>,
}

pub(super) async fn handle_channel_create(
    connection_id: &str,
    channel_id: i64,
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
    tx: &mpsc::Sender<OutboundMessage>,
) {
    match endpoint {
        "loadModel" => {
            let req: LoadModelChannelRequest = match serde_json::from_value(payload) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx
                        .send(OutboundMessage::ChannelError {
                            channel_id,
                            error: ProtocolError::with_cause(
                                "Invalid request",
                                format!("Invalid loadModel payload: {}", e),
                            ),
                        })
                        .await;
                    return;
                }
            };

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
                run_load_model_channel(connection_id, channel_id, req, state, tx, cancel).await;
            });
        }
        "getOrLoad" => {
            let req: GetOrLoadChannelRequest = match serde_json::from_value(payload) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx
                        .send(OutboundMessage::ChannelError {
                            channel_id,
                            error: ProtocolError::with_cause(
                                "Invalid request",
                                format!("Invalid getOrLoad payload: {}", e),
                            ),
                        })
                        .await;
                    return;
                }
            };

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
                run_get_or_load_channel(connection_id, channel_id, req, state, tx, cancel).await;
            });
        }
        "predict" => {
            let req: PredictChannelRequest = match serde_json::from_value(payload) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx
                        .send(OutboundMessage::ChannelError {
                            channel_id,
                            error: ProtocolError::with_cause(
                                "Invalid request",
                                format!("Invalid predict payload: {}", e),
                            ),
                        })
                        .await;
                    return;
                }
            };

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
                run_predict_channel(connection_id, channel_id, req, state, tx, cancel).await;
            });
        }
        _ => {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause(
                        "Unknown endpoint",
                        format!(
                            "Unknown {} channel endpoint: {}",
                            ApiNamespace::Llm.as_str(),
                            endpoint
                        ),
                    ),
                })
                .await;
        }
    }
}

// ============================================================================
// loadModel
// ============================================================================

async fn run_load_model_channel(
    connection_id: String,
    channel_id: i64,
    req: LoadModelChannelRequest,
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

    let resolved_info = llm_info_for_model_key(&req.model_key);
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "resolved", "info": resolved_info })),
            ack_id: None,
        })
        .await;

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "progress", "progress": 0.0 })),
            ack_id: None,
        })
        .await;

    let identifier = req
        .identifier
        .unwrap_or_else(|| default_identifier_from_key(&req.model_key));

    let mut config = LoadConfig {
        context_length: None,
        gpu_offload: GpuOffload::Max,
        ttl: req
            .ttl_ms
            .and_then(|ms| u64::try_from(ms).ok())
            .map(Duration::from_millis),
        no_hup: false,
    };
    apply_load_config_stack(&mut config, &req.load_config_stack);

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    state
        .signal_manager
        .update_loading_model(Some(&identifier), Some(0.1));
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "progress", "progress": 0.1 })),
            ack_id: None,
        })
        .await;

    match state
        .model_manager
        .load(&identifier, &req.model_key, config)
    {
        Ok(info) => {
            state.signal_manager.clear_loading();
            let loaded_ids = state
                .model_manager
                .list_loaded()
                .iter()
                .map(|m| m.identifier.clone())
                .collect();
            state.signal_manager.update_loaded_models(loaded_ids);

            let instance_info = llm_instance_info_from_full(&info, &req.model_key, req.ttl_ms);
            let _ = tx
                .send(OutboundMessage::ChannelSend {
                    channel_id,
                    message: Some(json!({ "type": "success", "info": instance_info })),
                    ack_id: None,
                })
                .await;
            let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        }
        Err(e) => {
            state.signal_manager.clear_loading();
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause("Load failed", e.to_string()),
                })
                .await;
        }
    }

    cleanup();
}

// ============================================================================
// getOrLoad
// ============================================================================

async fn run_get_or_load_channel(
    connection_id: String,
    channel_id: i64,
    req: GetOrLoadChannelRequest,
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

    let model_key = req.identifier;

    if let Some(model) = state.model_manager.get(&model_key) {
        model.touch().await;
        let instance_info = llm_instance_info_from_full(
            &model.info,
            &model_key,
            model.ttl.map(|d| d.as_millis() as i64),
        );
        let _ = tx
            .send(OutboundMessage::ChannelSend {
                channel_id,
                message: Some(json!({ "type": "alreadyLoaded", "info": instance_info })),
                ack_id: None,
            })
            .await;
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    let resolved_info = llm_info_for_model_key(&model_key);
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(
                json!({ "type": "startLoading", "identifier": model_key, "info": resolved_info }),
            ),
            ack_id: None,
        })
        .await;

    let mut config = LoadConfig {
        context_length: None,
        gpu_offload: GpuOffload::Max,
        ttl: req
            .load_ttl_ms
            .and_then(|ms| u64::try_from(ms).ok())
            .map(Duration::from_millis),
        no_hup: false,
    };
    apply_load_config_stack(&mut config, &req.load_config_stack);

    state
        .signal_manager
        .update_loading_model(Some(&model_key), Some(0.1));
    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "loadProgress", "progress": 0.1 })),
            ack_id: None,
        })
        .await;

    if cancel.is_cancelled() {
        let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        cleanup();
        return;
    }

    match state.model_manager.load(&model_key, &model_key, config) {
        Ok(info) => {
            state.signal_manager.clear_loading();
            let loaded_ids = state
                .model_manager
                .list_loaded()
                .iter()
                .map(|m| m.identifier.clone())
                .collect();
            state.signal_manager.update_loaded_models(loaded_ids);

            let instance_info = llm_instance_info_from_full(&info, &model_key, req.load_ttl_ms);
            let _ = tx
                .send(OutboundMessage::ChannelSend {
                    channel_id,
                    message: Some(json!({ "type": "loadSuccess", "info": instance_info })),
                    ack_id: None,
                })
                .await;
            let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;
        }
        Err(e) => {
            state.signal_manager.clear_loading();
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause("Load failed", e.to_string()),
                })
                .await;
        }
    }

    cleanup();
}

// ============================================================================
// predict
// ============================================================================

async fn run_predict_channel(
    connection_id: String,
    channel_id: i64,
    req: PredictChannelRequest,
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

    let _ = req.ignore_server_session_config;

    let prediction_config = kv_config_from_stack(&req.prediction_config_stack);
    let prediction_map = collapse_kv_stack(&req.prediction_config_stack);
    let mut settings = predict_settings_from_kv(&prediction_map);

    if let Some(ref preset_name) = req.fuzzy_preset_identifier {
        if let Some(preset) = SamplingPreset::from_name(preset_name) {
            let p = SamplingOptions::from_preset(preset);
            if !prediction_map.contains_key("temperature") {
                settings.temperature = p.temperature;
            }
            if !prediction_map.contains_key("topKSampling") {
                settings.top_k = p.top_k;
            }
            if !prediction_map.contains_key("topPSampling") {
                settings.top_p = p.top_p;
            }
            if !prediction_map.contains_key("minPSampling") {
                settings.min_p = p.min_p;
            }
            if !prediction_map.contains_key("repeatPenalty") {
                settings.repetition_penalty = p.repetition_penalty;
            }
        }
    }

    let model = match resolve_loaded_model(&req.model_specifier, &state) {
        Some(m) => m,
        None => {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause(
                        "Model not found",
                        "No loaded model matches the specifier",
                    ),
                })
                .await;
            cleanup();
            return;
        }
    };

    model.touch().await;

    let template = model.tokenizer.chat_template().ok().flatten();
    let bos_token = model.tokenizer.bos_token_string();
    let eos_token = model.tokenizer.eos_token_string();
    let messages = crate::common::chat_history_to_simple_messages(&req.history);
    let (prompt, template_stop_strings) = crate::server::rpc::format_messages_with_defaults(
        &messages,
        template.as_deref(),
        &model.info.architecture,
        Some(model.info.display_name.as_str()),
        bos_token,
        eos_token,
        true,
    );

    let mut all_stop_strings = settings.stop_strings;
    all_stop_strings.extend(template_stop_strings);

    let tokens = match model.tokenizer.encode(&prompt, true, false) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause("Tokenization failed", e.message),
                })
                .await;
            cleanup();
            return;
        }
    };

    let prompt_tokens_count = tokens.len();

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({ "type": "promptProcessingProgress", "progress": 1.0 })),
            ack_id: None,
        })
        .await;

    let mut stop_reason = "maxPredictedTokensReached".to_string();
    let mut first_token_time: Option<Instant> = None;
    let start_time = Instant::now();

    let max_new_tokens = if let Some(max_tokens) = settings.max_new_tokens {
        max_tokens
    } else {
        model
            .info
            .max_context_length
            .saturating_sub(prompt_tokens_count)
            .max(1)
    };

    let mut stop_tokens: Vec<i32> = Vec::new();
    for stop in &all_stop_strings {
        if let Ok(Some(id)) = model.tokenizer.piece_to_token(stop) {
            if !stop_tokens.contains(&id) {
                stop_tokens.push(id);
            }
        }
    }
    if let Ok(special) = model.tokenizer.special_tokens() {
        if let Some(eos_id) = special.eos_id {
            if !stop_tokens.contains(&eos_id) {
                stop_tokens.push(eos_id);
            }
        }
    }

    let gen_opts = GenerateOptions {
        max_new_tokens,
        stop_on_eos: true,
        stop_tokens,
    };

    let sampling = SamplingOptions {
        seed: settings.seed.unwrap_or(42),
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
        min_p: settings.min_p,
        repetition_penalty: settings.repetition_penalty,
        suppress_special_tokens: true,
    };

    let request_id = match {
        let mut engine = model.serving_engine.lock().await;
        engine.submit(&tokens, &gen_opts, &sampling)
    } {
        Ok(id) => id,
        Err(e) => {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause("Engine submit failed", e.message),
                })
                .await;
            cleanup();
            return;
        }
    };

    let mut generated = String::new();
    let mut sent_tokens = 0;
    let mut cancel_requested = false;

    loop {
        if !cancel_requested && cancel.is_cancelled() {
            cancel_requested = true;
            stop_reason = "userStopped".to_string();
            let _ = model.serving_engine.lock().await.cancel(request_id);
        }

        let (state, tokens_result) = {
            let mut engine = model.serving_engine.lock().await;
            let step_result = engine.step(1);
            let state = engine.request_state(request_id);
            let tokens = engine.get_tokens(request_id);
            (step_result.map(|_| state), tokens)
        };

        match state {
            Ok(RequestState::Decoding) | Ok(RequestState::Prefill) => {
                if let Ok(all_tokens) = tokens_result.as_ref() {
                    if all_tokens.len() > sent_tokens {
                        if first_token_time.is_none() && !all_tokens.is_empty() {
                            first_token_time = Some(Instant::now());
                        }

                        let new_tokens = &all_tokens[sent_tokens..];
                        if let Ok(text) = model.tokenizer.decode(new_tokens) {
                            if !text.is_empty() {
                                generated.push_str(&text);
                                if all_stop_strings.iter().any(|s| generated.contains(s)) {
                                    stop_reason = "stopStringFound".to_string();
                                }

                                let fragment = json!({
                                    "content": text,
                                    "tokensCount": i64::try_from(new_tokens.len()).unwrap_or(0),
                                    "containsDrafted": false,
                                    "reasoningType": "none",
                                    "isStructural": false,
                                });
                                let _ = tx
                                    .send(OutboundMessage::ChannelSend {
                                        channel_id,
                                        message: Some(
                                            json!({ "type": "fragment", "fragment": fragment }),
                                        ),
                                        ack_id: None,
                                    })
                                    .await;

                                if stop_reason == "stopStringFound" {
                                    break;
                                }
                            }
                        }
                        sent_tokens = all_tokens.len();
                    }
                }
                tokio::task::yield_now().await;
            }
            Ok(RequestState::Done) => {
                stop_reason = "eosFound".to_string();
                break;
            }
            Ok(RequestState::Canceled) => {
                if stop_reason != "userStopped" {
                    stop_reason = "userStopped".to_string();
                }
                break;
            }
            Ok(RequestState::Failed) => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::new("Generation failed"),
                    })
                    .await;
                let _ = model.serving_engine.lock().await.release(request_id);
                cleanup();
                return;
            }
            Err(e) => {
                let _ = tx
                    .send(OutboundMessage::ChannelError {
                        channel_id,
                        error: ProtocolError::with_cause("Engine step failed", e.message),
                    })
                    .await;
                let _ = model.serving_engine.lock().await.release(request_id);
                cleanup();
                return;
            }
            _ => {
                tokio::task::yield_now().await;
            }
        }

        if let Err(e) = tokens_result {
            let _ = tx
                .send(OutboundMessage::ChannelError {
                    channel_id,
                    error: ProtocolError::with_cause("Engine token fetch failed", e.message),
                })
                .await;
            let _ = model.serving_engine.lock().await.release(request_id);
            cleanup();
            return;
        }
    }

    let _ = model.serving_engine.lock().await.release(request_id);

    let total_time = start_time.elapsed();
    let time_to_first = first_token_time
        .map(|t| t.duration_since(start_time).as_secs_f64())
        .unwrap_or(0.0);
    let tokens_per_second = if total_time.as_secs_f64() > 0.0 {
        sent_tokens as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    let load_model_config = crate::protocol::types::KvConfig::empty()
        .with_field("contextLength", model.info.context_length);
    let instance_info = llm_instance_info_from_full(
        &model.info,
        &model.info.identifier,
        model.ttl.map(|d| d.as_millis() as i64),
    );

    let stats = json!({
        "stopReason": stop_reason,
        "tokensPerSecond": tokens_per_second,
        "timeToFirstTokenSec": time_to_first,
        "totalTimeSec": total_time.as_secs_f64(),
        "promptTokensCount": prompt_tokens_count,
        "predictedTokensCount": sent_tokens,
        "totalTokensCount": prompt_tokens_count + sent_tokens,
    });

    let _ = tx
        .send(OutboundMessage::ChannelSend {
            channel_id,
            message: Some(json!({
                "type": "success",
                "stats": stats,
                "modelInfo": instance_info,
                "loadModelConfig": load_model_config,
                "predictionConfig": prediction_config,
            })),
            ack_id: None,
        })
        .await;
    let _ = tx.send(OutboundMessage::ChannelClose { channel_id }).await;

    cleanup();
}

struct PredictSettings {
    max_new_tokens: Option<usize>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
}

fn predict_settings_from_kv(kv: &std::collections::BTreeMap<String, Value>) -> PredictSettings {
    let temperature = kv.get("temperature").and_then(Value::as_f64).unwrap_or(0.3) as f32;

    let max_new_tokens = kv
        .get("maxPredictedTokens")
        .and_then(checkbox_numeric_i64)
        .and_then(|v| v.and_then(|n| usize::try_from(n).ok()));

    let top_k = kv
        .get("topKSampling")
        .and_then(Value::as_i64)
        .and_then(|v| usize::try_from(v.max(0)).ok())
        .unwrap_or(30);

    let top_p = kv
        .get("topPSampling")
        .and_then(checkbox_numeric_f64)
        .flatten()
        .unwrap_or(0.9) as f32;

    let min_p = kv
        .get("minPSampling")
        .and_then(checkbox_numeric_f64)
        .flatten()
        .unwrap_or(0.0) as f32;

    let repetition_penalty = kv
        .get("repeatPenalty")
        .and_then(checkbox_numeric_f64)
        .flatten()
        .unwrap_or(1.05) as f32;

    let seed = kv
        .get("seed")
        .and_then(checkbox_numeric_i64)
        .flatten()
        .and_then(|v| u64::try_from(v.max(0)).ok());

    let stop_strings = kv
        .get("stopStrings")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    PredictSettings {
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
        stop_strings,
    }
}

fn checkbox_numeric_i64(value: &Value) -> Option<Option<i64>> {
    if let Some(obj) = value.as_object() {
        let checked = obj.get("checked")?.as_bool()?;
        if !checked {
            return Some(None);
        }
        return obj.get("value").and_then(Value::as_i64).map(Some);
    }
    value.as_i64().map(Some)
}

fn checkbox_numeric_f64(value: &Value) -> Option<Option<f64>> {
    if let Some(obj) = value.as_object() {
        let checked = obj.get("checked")?.as_bool()?;
        if !checked {
            return Some(None);
        }
        return obj.get("value").and_then(Value::as_f64).map(Some);
    }
    value.as_f64().map(Some)
}
