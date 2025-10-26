use std::sync::Arc;

use serde_json::{json, Value};

use crate::protocol::messages::ProtocolError;
use crate::protocol::types::KvConfig;
use crate::server::websocket::ServerState;

pub(super) async fn handle_rpc(
    endpoint: &str,
    payload: Value,
    state: &Arc<ServerState>,
) -> Result<Option<Value>, ProtocolError> {
    match endpoint {
        "unloadModel" => {
            let req: super::types::UnloadModelRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            state
                .model_manager
                .unload(&req.identifier)
                .map_err(|e| ProtocolError::with_cause("Unload failed", e.to_string()))?;
            Ok(None)
        }
        "listLoaded" => {
            let loaded = state.model_manager.list_loaded();
            let models = loaded
                .iter()
                .map(super::loaded_model_to_llm_instance_info)
                .collect::<Vec<_>>();
            Ok(Some(Value::Array(models)))
        }
        "getModelInfo" => {
            let req: super::types::GetModelInfoRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state);
            if model.is_none() && req.throw_if_not_found {
                return Err(ProtocolError::with_cause(
                    "Model not found",
                    "No loaded model matches the specifier",
                ));
            }
            Ok(model.map(|m| super::loaded_model_to_llm_instance_info(&m.info)))
        }
        "getLoadConfig" => {
            let req: super::types::GetLoadConfigRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state)
                .ok_or_else(|| {
                    ProtocolError::with_cause(
                        "Model not found",
                        "No loaded model matches the specifier",
                    )
                })?;
            let config = KvConfig::empty().with_field("contextLength", model.info.context_length);
            Ok(Some(serde_json::to_value(config).unwrap_or(Value::Null)))
        }
        "getBasePredictionConfig" => {
            let req: super::types::GetBasePredictionConfigRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = crate::server::model_utils::resolve_loaded_model(&req.specifier, state);
            let config = KvConfig::empty()
                .with_field("temperature", 0.8)
                .with_field(
                    "maxPredictedTokens",
                    json!({ "checked": false, "value": 1000 }),
                )
                .with_field("stopStrings", Vec::<String>::new())
                .with_field("topKSampling", 40)
                .with_field("repeatPenalty", json!({ "checked": true, "value": 1.1 }))
                .with_field("minPSampling", json!({ "checked": true, "value": 0.05 }))
                .with_field("topPSampling", json!({ "checked": true, "value": 0.95 }))
                .with_field("seed", json!({ "checked": false, "value": -1 }));
            Ok(Some(serde_json::to_value(config).unwrap_or(Value::Null)))
        }
        "getInstanceProcessingState" => {
            let req: super::types::GetInstanceProcessingStateRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state);
            if model.is_none() && req.throw_if_not_found {
                return Err(ProtocolError::with_cause(
                    "Model not found",
                    "No loaded model matches the specifier",
                ));
            }
            Ok(Some(json!({ "status": "idle", "queued": 0 })))
        }
        "estimateModelUsage" => {
            let req: super::types::EstimateModelUsageRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _load_config = crate::common::collapse_kv_stack(&req.load_config_stack);
            Ok(Some(super::estimate_model_usage(&req.model_key)))
        }
        "applyPromptTemplate" => {
            let req: super::types::ApplyPromptTemplateRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _prediction_config = crate::common::collapse_kv_stack(&req.prediction_config_stack);
            let omit_bos = req.opts.omit_bos_token.unwrap_or(false);
            let omit_eos = req.opts.omit_eos_token.unwrap_or(false);
            let _ = (
                omit_bos,
                omit_eos,
                req.opts.tool_definitions.as_ref().map(Vec::len),
            );
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state)
                .ok_or_else(|| {
                    ProtocolError::with_cause(
                        "Model not found",
                        "No loaded model matches the specifier",
                    )
                })?;
            model.touch().await;

            let template = model.tokenizer.chat_template().ok().flatten();
            let messages = crate::common::chat_history_to_simple_messages(&req.history);
            let bos_token = if omit_bos {
                None
            } else {
                model.tokenizer.bos_token_string()
            };
            let eos_token = if omit_eos {
                None
            } else {
                model.tokenizer.eos_token_string()
            };
            let (formatted, _stop_strings) = super::format_messages_with_defaults(
                &messages,
                template.as_deref(),
                &model.info.architecture,
                Some(model.info.display_name.as_str()),
                bos_token,
                eos_token,
                true,
            );
            Ok(Some(json!({ "formatted": formatted })))
        }
        "tokenize" => {
            let req: super::types::TokenizeRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state)
                .ok_or_else(|| {
                    ProtocolError::with_cause(
                        "Model not found",
                        "No loaded model matches the specifier",
                    )
                })?;
            model.touch().await;
            let tokens = model
                .tokenizer
                .encode(&req.input_string, false, false)
                .map_err(|e| ProtocolError::with_cause("Tokenize failed", e.message))?;
            Ok(Some(json!({ "tokens": tokens })))
        }
        "countTokens" => {
            let req: super::types::CountTokensRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let model = crate::server::model_utils::resolve_loaded_model(&req.specifier, state)
                .ok_or_else(|| {
                    ProtocolError::with_cause(
                        "Model not found",
                        "No loaded model matches the specifier",
                    )
                })?;
            model.touch().await;
            let tokens = model
                .tokenizer
                .encode(&req.input_string, false, false)
                .map_err(|e| ProtocolError::with_cause("Tokenize failed", e.message))?;
            Ok(Some(json!({ "tokenCount": tokens.len() })))
        }
        "preloadDraftModel" => {
            let req: super::types::PreloadDraftModelRequest =
                super::parse_payload(payload).map_err(super::invalid_request)?;
            let _ = (&req.specifier, &req.draft_model_key);
            Ok(None)
        }
        _ => Err(ProtocolError::with_suggestion(
            "Endpoint not found",
            format!("Unknown llm RPC endpoint: {}", endpoint),
            "Check the endpoint name or SDK version",
        )),
    }
}
