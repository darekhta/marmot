use std::time::Duration;

use serde_json::{json, Value};

use crate::common::{collapse_kv_stack, quantization_to_value, unix_time_ms};
use crate::config::ModelResolver;
use crate::engine::model_manager::{FullModelInfo, GpuOffload, LoadConfig};
use crate::protocol::types::{KvConfig, KvConfigField, KvConfigStack};

pub(super) fn default_identifier_from_key(model_key: &str) -> String {
    std::path::Path::new(model_key)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| model_key.to_string())
}

pub(super) fn llm_info_for_model_key(model_key: &str) -> Value {
    let resolver = ModelResolver::new();
    let resolved_path = std::fs::metadata(model_key)
        .ok()
        .map(|_| model_key.to_string())
        .or_else(|| {
            resolver
                .resolve(model_key)
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        });

    let display_name = resolved_path
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_stem().and_then(|s| s.to_str()))
        .unwrap_or(model_key)
        .to_string();

    let quant = resolved_path
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_name().and_then(|s| s.to_str()))
        .and_then(|name| {
            [
                "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K", "Q2_K",
                "F16", "F32", "BF16",
            ]
            .iter()
            .find(|pat| name.contains(**pat))
            .copied()
        });

    let size_bytes = resolved_path
        .as_deref()
        .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .unwrap_or(0);

    json!({
        "type": "llm",
        "modelKey": model_key,
        "format": "gguf",
        "displayName": display_name,
        "publisher": "local",
        "path": model_key,
        "sizeBytes": size_bytes,
        "quantization": quantization_to_value(quant),
        "vision": false,
        "trainedForToolUse": false,
        "maxContextLength": 2048,
    })
}

pub(super) fn llm_instance_info_from_full(
    info: &FullModelInfo,
    model_key: &str,
    ttl_ms: Option<i64>,
) -> Value {
    json!({
        "type": "llm",
        "modelKey": model_key,
        "format": info.format,
        "displayName": info.display_name,
        "publisher": "local",
        "path": model_key,
        "sizeBytes": info.size_bytes,
        "paramsString": info.params_string,
        "architecture": info.architecture,
        "quantization": quantization_to_value(info.quantization.as_deref()),
        "identifier": info.identifier,
        "instanceReference": format!("instance:{}", info.identifier),
        "ttlMs": ttl_ms,
        "lastUsedTime": unix_time_ms(),
        "vision": info.vision,
        "trainedForToolUse": info.trained_for_tool_use,
        "maxContextLength": info.max_context_length,
        "contextLength": info.context_length,
    })
}

pub(super) fn kv_config_from_stack(stack: &KvConfigStack) -> KvConfig {
    let collapsed = collapse_kv_stack(stack);
    KvConfig {
        fields: collapsed
            .into_iter()
            .map(|(key, value)| KvConfigField { key, value })
            .collect(),
    }
}

pub(super) fn apply_load_config_stack(config: &mut LoadConfig, stack: &KvConfigStack) {
    let collapsed = collapse_kv_stack(stack);
    if let Some(v) = collapsed.get("contextLength").and_then(Value::as_u64) {
        config.context_length = usize::try_from(v).ok();
    }

    if let Some(v) = collapsed.get("gpuOffload").and_then(Value::as_str) {
        config.gpu_offload = match v {
            "off" => GpuOffload::Off,
            _ => GpuOffload::Max,
        };
    }

    if let Some(v) = collapsed.get("ttlMs").and_then(Value::as_u64) {
        config.ttl = Some(Duration::from_millis(v));
    } else if let Some(v) = collapsed.get("ttl").and_then(Value::as_u64) {
        config.ttl = Some(Duration::from_secs(v));
    }

    if let Some(v) = collapsed.get("noHup").and_then(Value::as_bool) {
        config.no_hup = v;
    }
}
