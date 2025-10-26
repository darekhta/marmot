use std::sync::Arc;

use crate::engine::model_manager::LoadedModel;
use crate::protocol::types::ModelSpecifier;
use crate::server::websocket::ServerState;

pub fn resolve_loaded_model(
    specifier: &ModelSpecifier,
    state: &Arc<ServerState>,
) -> Option<Arc<LoadedModel>> {
    match specifier {
        ModelSpecifier::InstanceReference { instance_reference } => {
            let identifier = instance_reference
                .strip_prefix("instance:")
                .unwrap_or(instance_reference);
            state.model_manager.get(identifier)
        }
        ModelSpecifier::Query { query } => {
            if let Some(domain) = query.domain.as_deref() {
                if domain != "llm" && domain != "embedding" {
                    return None;
                }
            }
            if query.vision.unwrap_or(false) {
                return None;
            }
            if let Some(identifier) = query.identifier.as_deref() {
                return state.model_manager.get(identifier);
            }
            if let Some(path) = query.path.as_deref() {
                let loaded = state.model_manager.list_loaded();
                let identifier = loaded
                    .iter()
                    .find(|m| m.path == path)
                    .map(|m| m.identifier.clone())?;
                return state.model_manager.get(&identifier);
            }
            None
        }
    }
}
