//! Reactive signals for LM Studio SDK compatibility
//!
//! Implements the signal protocol for reactive state synchronization.
//! Clients can subscribe to signal paths and receive updates when state changes.

use std::collections::HashMap;

use parking_lot::RwLock;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::protocol::messages::{OutboundMessage, ProtocolError};

/// Known signal endpoints
pub mod endpoints {
    /// List of loaded model identifiers
    pub const MODELS_LOADED: &str = "models.loaded";
    /// Currently loading model info (identifier, progress)
    pub const MODELS_LOADING: &str = "models.loading";
    /// Server status
    pub const SERVER_STATUS: &str = "server.status";
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct SubscriptionKey {
    connection_id: String,
    subscribe_id: i64,
}

/// Signal subscription entry
struct Subscription {
    /// Signal endpoint being subscribed to
    endpoint: String,
    /// Sender to the subscribed client
    tx: mpsc::Sender<OutboundMessage>,
}

struct WritableSubscription {
    endpoint: String,
    tx: mpsc::Sender<OutboundMessage>,
}

/// Manages signal subscriptions and broadcasts
pub struct SignalManager {
    /// Map of subscribe_id -> subscription
    subscriptions: RwLock<HashMap<SubscriptionKey, Subscription>>,
    writable_subscriptions: RwLock<HashMap<SubscriptionKey, WritableSubscription>>,
    /// Current signal values (cached for new subscribers)
    values: RwLock<HashMap<String, Value>>,
}

impl SignalManager {
    pub fn new() -> Self {
        let mut values = HashMap::new();
        // Initialize default values
        values.insert(endpoints::MODELS_LOADED.to_string(), json!([]));
        values.insert(endpoints::MODELS_LOADING.to_string(), Value::Null);
        values.insert(endpoints::SERVER_STATUS.to_string(), json!("ready"));

        Self {
            subscriptions: RwLock::new(HashMap::new()),
            writable_subscriptions: RwLock::new(HashMap::new()),
            values: RwLock::new(values),
        }
    }

    /// Subscribe a connection to a signal endpoint
    pub fn subscribe(
        &self,
        endpoint: &str,
        subscribe_id: i64,
        connection_id: &str,
        _creation_parameter: Value,
        tx: mpsc::Sender<OutboundMessage>,
    ) {
        // Check if endpoint is valid
        let current_value = match self.get_value(endpoint) {
            Some(value) => value,
            None => {
                // Unknown endpoint - send signalError
                tracing::warn!(
                    "Connection {} tried to subscribe to unknown signal endpoint: {}",
                    connection_id,
                    endpoint
                );
                let _ = tx.try_send(OutboundMessage::SignalError {
                    subscribe_id,
                    error: ProtocolError::with_suggestion(
                        "Unknown signal endpoint",
                        format!("Signal endpoint '{}' does not exist", endpoint),
                        "Check available signal endpoints: models.loaded, models.loading, server.status",
                    ),
                });
                return;
            }
        };

        let mut subs = self.subscriptions.write();
        let key = SubscriptionKey {
            connection_id: connection_id.to_string(),
            subscribe_id,
        };

        // Check if already subscribed with this ID
        if subs.contains_key(&key) {
            tracing::warn!(
                "Connection {} tried to resubscribe with existing subscription ID: {}",
                connection_id,
                subscribe_id
            );
            let _ = tx.try_send(OutboundMessage::SignalError {
                subscribe_id,
                error: ProtocolError::new("Subscription ID already in use"),
            });
            return;
        }

        // Store subscription by subscribe_id
        subs.insert(
            key,
            Subscription {
                endpoint: endpoint.to_string(),
                tx: tx.clone(),
            },
        );

        tracing::debug!(
            "Connection {} subscribed to signal '{}' with id '{}'",
            connection_id,
            endpoint,
            subscribe_id
        );

        // Send initial value using JSON patch format (replace at root)
        let msg = OutboundMessage::SignalUpdate {
            subscribe_id,
            patches: vec![json!({
                "op": "replace",
                "path": [],
                "value": current_value
            })],
            tags: vec![],
        };
        let _ = tx.try_send(msg);
    }

    /// Unsubscribe by subscription ID
    pub fn unsubscribe_by_id(&self, connection_id: &str, subscribe_id: i64) {
        let mut subs = self.subscriptions.write();
        let key = SubscriptionKey {
            connection_id: connection_id.to_string(),
            subscribe_id,
        };
        if let Some((_, sub)) = subs.remove_entry(&key) {
            tracing::debug!(
                "Connection {} unsubscribed from signal '{}' (id: {})",
                connection_id,
                sub.endpoint,
                subscribe_id
            );
        }
    }

    /// Unsubscribe a connection from all signals (on disconnect)
    pub fn unsubscribe_all(&self, connection_id: &str) {
        let mut subs = self.subscriptions.write();
        subs.retain(|k, _| k.connection_id != connection_id);
        let mut writable_subs = self.writable_subscriptions.write();
        writable_subs.retain(|k, _| k.connection_id != connection_id);
        tracing::debug!("Connection {} unsubscribed from all signals", connection_id);
    }

    /// Update a signal value and broadcast to subscribers
    pub fn update(&self, endpoint: &str, value: Value) {
        // Update cached value
        {
            let mut values = self.values.write();
            values.insert(endpoint.to_string(), value.clone());
        }

        // Broadcast to subscribers of this endpoint
        self.broadcast(endpoint, value);
    }

    /// Broadcast a value to all subscribers of an endpoint
    fn broadcast(&self, endpoint: &str, value: Value) {
        let mut subs = self.subscriptions.write();

        for (key, sub) in subs.iter_mut() {
            if sub.endpoint == endpoint {
                // Create JSON patch for the update (simple replace at root)
                let msg = OutboundMessage::SignalUpdate {
                    subscribe_id: key.subscribe_id,
                    patches: vec![json!({
                        "op": "replace",
                        "path": [],
                        "value": value.clone()
                    })],
                    tags: vec![],
                };

                // Non-blocking send
                let _ = sub.tx.try_send(msg);
            }
        }
    }

    /// Get current value of a signal
    pub fn get_value(&self, endpoint: &str) -> Option<Value> {
        let values = self.values.read();
        values.get(endpoint).cloned()
    }

    /// Update models.loaded signal with current list
    pub fn update_loaded_models(&self, identifiers: Vec<String>) {
        self.update(endpoints::MODELS_LOADED, json!(identifiers));
    }

    /// Update models.loading signal
    pub fn update_loading_model(&self, identifier: Option<&str>, progress: Option<f32>) {
        let value = match (identifier, progress) {
            (Some(id), Some(p)) => json!({
                "identifier": id,
                "progress": p,
            }),
            (Some(id), None) => json!({
                "identifier": id,
                "progress": 0.0,
            }),
            _ => Value::Null,
        };
        self.update(endpoints::MODELS_LOADING, value);
    }

    /// Clear loading state
    pub fn clear_loading(&self) {
        self.update(endpoints::MODELS_LOADING, Value::Null);
    }

    pub fn subscribe_writable(
        &self,
        endpoint: &str,
        subscribe_id: i64,
        connection_id: &str,
        _creation_parameter: Value,
        tx: mpsc::Sender<OutboundMessage>,
    ) {
        let current_value = match self.get_value(endpoint) {
            Some(value) => value,
            None => {
                let _ = tx.try_send(OutboundMessage::WritableSignalError {
                    subscribe_id,
                    error: ProtocolError::with_suggestion(
                        "Unknown signal endpoint",
                        format!("Signal endpoint '{}' does not exist", endpoint),
                        "Check available signal endpoints: models.loaded, models.loading, server.status",
                    ),
                });
                return;
            }
        };

        let mut subs = self.writable_subscriptions.write();
        let key = SubscriptionKey {
            connection_id: connection_id.to_string(),
            subscribe_id,
        };
        if subs.contains_key(&key) {
            let _ = tx.try_send(OutboundMessage::WritableSignalError {
                subscribe_id,
                error: ProtocolError::new("Subscription ID already in use"),
            });
            return;
        }

        subs.insert(
            key,
            WritableSubscription {
                endpoint: endpoint.to_string(),
                tx: tx.clone(),
            },
        );

        let _ = tx.try_send(OutboundMessage::WritableSignalUpdate {
            subscribe_id,
            patches: vec![json!({
                "op": "replace",
                "path": [],
                "value": current_value,
            })],
            tags: vec![],
        });
    }

    pub fn unsubscribe_writable_by_id(&self, connection_id: &str, subscribe_id: i64) {
        let mut subs = self.writable_subscriptions.write();
        subs.remove(&SubscriptionKey {
            connection_id: connection_id.to_string(),
            subscribe_id,
        });
    }

    pub fn apply_writable_update(
        &self,
        connection_id: &str,
        subscribe_id: i64,
        patches: Vec<Value>,
        tags: Vec<String>,
    ) {
        let endpoint = {
            let subs = self.writable_subscriptions.read();
            let Some(sub) = subs.get(&SubscriptionKey {
                connection_id: connection_id.to_string(),
                subscribe_id,
            }) else {
                return;
            };
            sub.endpoint.clone()
        };

        let mut root_replace: Option<Value> = None;
        for patch in &patches {
            if patch.get("op").and_then(|v| v.as_str()) != Some("replace") {
                continue;
            }
            let Some(path) = patch.get("path").and_then(|v| v.as_array()) else {
                continue;
            };
            if !path.is_empty() {
                continue;
            }
            if let Some(value) = patch.get("value") {
                root_replace = Some(value.clone());
            }
        }

        if let Some(value) = root_replace {
            self.update(&endpoint, value);
        }

        let subs = self.writable_subscriptions.read();
        for (key, sub) in subs.iter() {
            if sub.endpoint != endpoint {
                continue;
            }
            let _ = sub.tx.try_send(OutboundMessage::WritableSignalUpdate {
                subscribe_id: key.subscribe_id,
                patches: patches.clone(),
                tags: tags.clone(),
            });
        }
    }
}

impl Default for SignalManager {
    fn default() -> Self {
        Self::new()
    }
}
