//! `marmot-lm run` command - run inference

use crate::client::api::{LoadModelEvent, PredictEvent};
use crate::client::{try_connect_server, WsClient};
use crate::config::ModelResolver;
use crate::embedded::EmbeddedRuntime;
use crate::ffi::SamplingOptions;
use crate::protocol::types::{
    ChatHistoryData, ChatMessageData, ChatMessagePartData, KvConfig, KvConfigField, KvConfigStack,
    KvConfigStackLayer, ModelQuery, ModelSpecifier,
};
use serde_json::json;
use std::io::Write;
use std::time::Duration;

const DEFAULT_PORT: u16 = 1234;

pub async fn run(model_spec: &str, prompt: Option<&str>, temperature: f32) -> anyhow::Result<()> {
    // Try connecting to server first
    match try_connect_server(DEFAULT_PORT, "llm").await {
        Some(client) => {
            tracing::info!("Connected to marmot-lm server on port {}", DEFAULT_PORT);
            run_via_server(client, model_spec, prompt, temperature).await
        }
        None => {
            tracing::info!("No server running, using embedded mode");
            run_embedded(model_spec, prompt, temperature).await
        }
    }
}

async fn run_via_server(
    client: WsClient,
    model_spec: &str,
    prompt: Option<&str>,
    temperature: f32,
) -> anyhow::Result<()> {
    if std::env::var_os("MARMOT_LM_SIGNAL_PROBE").is_some() {
        probe_client_signals(&client).await?;
    }

    // Check if model is loaded, if not load it
    let llm = client.llm();
    let loaded_models = llm.list_loaded().await?;

    // Try to find model by exact match or by checking if the identifier contains the spec
    // This handles both "tinyllama-1.1b-chat" and full paths
    let existing_identifier = loaded_models.iter().find_map(|m| {
        let id = m.identifier.as_str();
        if id == model_spec || model_spec.contains(id) || id.contains(model_spec) {
            Some(id.to_string())
        } else {
            None
        }
    });

    // Use existing identifier or load the model
    let model_identifier = if let Some(id) = existing_identifier {
        println!("Model already loaded: {}", id);
        id
    } else {
        println!("Loading model: {}", model_spec);
        let mut load_channel = llm.load_model(model_spec, empty_kv_config_stack()).await?;

        // Wait for load to complete and extract identifier
        let mut loaded_id = None;
        while let Some(event) = load_channel.recv().await {
            match event {
                LoadModelEvent::Resolved => {
                    print!("\rLoading: 0.0%");
                    std::io::stdout().flush()?;
                }
                LoadModelEvent::Progress { progress } => {
                    print!("\rLoading: {:.1}%", progress * 100.0);
                    std::io::stdout().flush()?;
                }
                LoadModelEvent::Success { info } => {
                    loaded_id = Some(info.identifier);
                    println!("\rLoading: Done     ");
                    break;
                }
                LoadModelEvent::Unknown => {}
            }
        }

        loaded_id.ok_or_else(|| anyhow::anyhow!("Failed to load model - no identifier returned"))?
    };

    if let Some(prompt_text) = prompt {
        // Single prompt mode - use predict channel
        let history = chat_history_with_user_message(prompt_text);
        let top_k = if temperature == 0.0 { 1 } else { 30 };
        let prediction_config_stack = make_prediction_config_stack(None, temperature as f64, top_k, 0.9);
        let model_specifier = model_specifier_for_identifier(&model_identifier);
        let mut channel = llm
            .predict(model_specifier, history, prediction_config_stack)
            .await?;

        // Stream responses
        let mut response = String::new();
        while let Some(event) = channel.recv().await {
            if let PredictEvent::Fragment(fragment) = event {
                response.push_str(&fragment.content);
                print!("{}", fragment.content);
                std::io::stdout().flush()?;
            }
        }
        println!();
    } else {
        // Interactive mode
        println!("Interactive mode via server. Type 'exit' to quit.\n");

        let stdin = std::io::stdin();
        let mut messages: Vec<ChatMessageData> = Vec::new();
        loop {
            print!("> ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            stdin.read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }
            if input == "exit" || input == "quit" {
                break;
            }

            messages.push(chat_message("user", input));
            let history = ChatHistoryData {
                messages: messages.clone(),
            };
            let top_k = if temperature == 0.0 { 1 } else { 30 };
            let prediction_config_stack = make_prediction_config_stack(None, temperature as f64, top_k, 0.9);
            let model_specifier = model_specifier_for_identifier(&model_identifier);
            let mut channel = llm
                .predict(model_specifier, history, prediction_config_stack)
                .await?;

            // Stream responses
            let mut response = String::new();
            while let Some(event) = channel.recv().await {
                if let PredictEvent::Fragment(fragment) = event {
                    response.push_str(&fragment.content);
                    print!("{}", fragment.content);
                    std::io::stdout().flush()?;
                }
            }
            messages.push(chat_message("assistant", &response));
            println!("\n");
        }
    }

    client.close().await?;
    Ok(())
}

async fn run_embedded(model_spec: &str, prompt: Option<&str>, temperature: f32) -> anyhow::Result<()> {
    // Resolve the model specifier to a path
    let resolver = ModelResolver::new();
    let model_path = resolver.resolve(model_spec)?;
    let model_path_str = model_path.to_string_lossy();

    tracing::info!("Loading model from: {}", model_path_str);

    let mut runtime = EmbeddedRuntime::new(&model_path_str)?;
    let info = runtime.model_info()?;
    tracing::info!(
        "Loaded model: {} ({}L, {}H, vocab={})",
        info.architecture,
        info.n_layer,
        info.n_head,
        info.n_vocab
    );

    if let Some(prompt) = prompt {
        // Single prompt mode
        let greedy = temperature == 0.0;
        let sampling = SamplingOptions {
            temperature,
            top_k: if greedy { 1 } else { 30 },
            suppress_special_tokens: !greedy,
            ..Default::default()
        };
        let _ = runtime.generate_streaming(prompt, 0, &sampling)?;
        println!();
    } else {
        // Interactive mode
        runtime.run_interactive(None)?;
    }

    Ok(())
}

async fn probe_client_signals(client: &WsClient) -> anyhow::Result<()> {
    client.send_keepalive().await?;

    let mut status_signal = client.subscribe_signal("server.status").await?;
    let _ = tokio::time::timeout(Duration::from_secs(1), status_signal.recv()).await;
    client.unsubscribe_signal(status_signal.id()).await?;

    let mut writable_signal = client.subscribe_writable_signal("server.status").await?;
    let _ = tokio::time::timeout(Duration::from_secs(1), writable_signal.recv()).await;
    client
        .update_writable_signal(writable_signal.id(), Vec::new(), Vec::new())
        .await?;
    client
        .unsubscribe_writable_signal(writable_signal.id())
        .await?;

    client
        .send_communication_warning("client signal probe")
        .await?;

    let system_ws = WsClient::connect(DEFAULT_PORT, "system").await?;
    let system_version = system_ws.system().version().await?;
    let _ = system_version.version.as_str();
    let _ = system_version.build;
    system_ws.close().await?;

    Ok(())
}

fn chat_history_with_user_message(message: &str) -> ChatHistoryData {
    ChatHistoryData {
        messages: vec![chat_message("user", message)],
    }
}

fn chat_message(role: &str, content: &str) -> ChatMessageData {
    ChatMessageData {
        role: role.to_string(),
        content: vec![ChatMessagePartData::Text {
            text: content.to_string(),
        }],
    }
}

fn make_prediction_config_stack(
    max_predicted_tokens: Option<i64>,
    temperature: f64,
    top_k_sampling: i64,
    top_p_sampling: f64,
) -> KvConfigStack {
    let mut fields = Vec::new();
    if let Some(max_predicted_tokens) = max_predicted_tokens {
        fields.push(KvConfigField {
            key: "maxPredictedTokens".to_string(),
            value: json!(max_predicted_tokens),
        });
    }
    fields.push(KvConfigField {
        key: "temperature".to_string(),
        value: json!(temperature),
    });
    fields.push(KvConfigField {
        key: "topKSampling".to_string(),
        value: json!(top_k_sampling),
    });
    fields.push(KvConfigField {
        key: "topPSampling".to_string(),
        value: json!(top_p_sampling),
    });

    KvConfigStack {
        layers: vec![KvConfigStackLayer {
            layer_name: "apiOverride".to_string(),
            config: KvConfig { fields },
        }],
    }
}

fn empty_kv_config_stack() -> KvConfigStack {
    KvConfigStack { layers: Vec::new() }
}

fn model_specifier_for_identifier(identifier: &str) -> ModelSpecifier {
    ModelSpecifier::Query {
        query: ModelQuery {
            domain: None,
            identifier: Some(identifier.to_string()),
            path: None,
            vision: None,
        },
    }
}
