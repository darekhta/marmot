//! Shared helpers for server modules.

use std::collections::BTreeMap;
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};

use crate::protocol::types::{ChatHistoryData, KvConfigStack};

#[derive(Debug, Clone)]
pub struct SimpleMessage {
    pub role: String,
    pub content: String,
}

pub fn unix_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn quantization_bits(name: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix('Q') {
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(bits) = digits.parse::<i64>() {
            return Some(bits);
        }
    }
    if let Some(rest) = name.strip_prefix('F') {
        if let Ok(bits) = rest.parse::<i64>() {
            return Some(bits);
        }
    }
    None
}

pub fn quantization_to_value(name: Option<&str>) -> Option<Value> {
    let name = name?;
    let bits = quantization_bits(name)?;
    Some(json!({ "name": name, "bits": bits }))
}

pub fn collapse_kv_stack(stack: &KvConfigStack) -> BTreeMap<String, Value> {
    let mut result = BTreeMap::<String, Value>::new();
    for layer in &stack.layers {
        let _ = &layer.layer_name;
        for field in &layer.config.fields {
            result.insert(field.key.clone(), field.value.clone());
        }
    }
    result
}

pub fn chat_history_to_simple_messages(history: &ChatHistoryData) -> Vec<SimpleMessage> {
    history
        .messages
        .iter()
        .map(|m| SimpleMessage {
            role: m.role.clone(),
            content: m
                .content
                .iter()
                .map(|p| p.to_prompt_fragment())
                .collect::<String>(),
        })
        .collect()
}
