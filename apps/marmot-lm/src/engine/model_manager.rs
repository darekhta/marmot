//! Model loading and management

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::Serialize;
use tokio::sync::{Mutex, RwLock};

use crate::config::{scan_hf_cache, scan_models_dir, ModelResolver};
use crate::ffi::{Backend, Context, Model, ServingEngine, Tokenizer};

/// Load configuration options
#[derive(Debug, Clone, Default)]
pub struct LoadConfig {
    /// Context length override (None = use model default)
    pub context_length: Option<usize>,
    /// GPU offload mode (off => CPU backend)
    pub gpu_offload: GpuOffload,
    /// Time-to-live before auto-unload (None = never)
    pub ttl: Option<Duration>,
    /// Keep model loaded after client disconnects
    pub no_hup: bool,
}

/// GPU offload configuration
#[derive(Debug, Clone, Default)]
pub enum GpuOffload {
    #[default]
    Max,
    Off,
}

/// Full model information for API responses
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FullModelInfo {
    pub identifier: String,
    pub path: String,
    pub architecture: String,
    pub context_length: usize,
    pub max_context_length: usize,
    pub display_name: String,
    pub format: String,
    pub quantization: Option<String>,
    pub params_string: String,
    pub size_bytes: u64,
    #[serde(rename = "type")]
    pub model_type: String,
    pub vision: bool,
    pub trained_for_tool_use: bool,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
}

/// Downloaded model entry (not loaded)
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadedModelInfo {
    pub path: String,
    pub name: String,
    pub size_bytes: u64,
    pub quantization: Option<String>,
}

/// A loaded model with its context, tokenizer, and runtime state
pub struct LoadedModel {
    pub identifier: String,
    pub tokenizer: Tokenizer,
    pub serving_engine: Mutex<ServingEngine>,
    // Held for RAII: dropping triggers C cleanup (marmot_model_destroy / marmot_destroy)
    #[allow(dead_code)]
    pub model: Model,
    #[allow(dead_code)]
    pub ctx: Context,
    /// Cached model info
    pub info: FullModelInfo,
    /// Last time the model was used
    pub last_used: RwLock<Instant>,
    /// TTL for auto-unload
    pub ttl: Option<Duration>,
    /// Keep loaded after client disconnect
    pub no_hup: bool,
}

impl LoadedModel {
    /// Update last used timestamp
    pub async fn touch(&self) {
        *self.last_used.write().await = Instant::now();
    }
}

/// Manages loaded models with LM Studio SDK compatibility
pub struct ModelManager {
    models: DashMap<String, Arc<LoadedModel>>,
    resolver: ModelResolver,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
            resolver: ModelResolver::new(),
        }
    }

    /// Load a model from a file path or model specifier
    pub fn load(
        &self,
        identifier: &str,
        path_or_spec: &str,
        config: LoadConfig,
    ) -> anyhow::Result<FullModelInfo> {
        // Check if already loaded
        if self.models.contains_key(identifier) {
            return Err(anyhow::anyhow!("Model '{}' is already loaded", identifier));
        }

        // Resolve path
        let path = self.resolver.resolve(path_or_spec)?;
        let path_str = path.to_string_lossy();

        tracing::info!("Loading model '{}' from {}", identifier, path_str);

        let backend = match config.gpu_offload {
            GpuOffload::Off => Backend::Cpu,
            GpuOffload::Max => Backend::Metal,
        };
        let ctx = Context::new(backend)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;
        let model =
            Model::load(&path).map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        let tokenizer = Tokenizer::from_gguf(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let model_info = model
            .info()
            .map_err(|e| anyhow::anyhow!("Failed to get model info: {}", e))?;

        // Get file size
        let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        // Extract quantization from filename
        let quantization = extract_quantization(&path);

        // Build display name
        let display_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(identifier)
            .to_string();

        // Estimate params from architecture
        let params_string = estimate_params(
            &model_info.architecture,
            model_info.n_layer,
            model_info.n_embd,
        );

        let full_info = FullModelInfo {
            identifier: identifier.to_string(),
            path: path_str.to_string(),
            architecture: model_info.architecture.clone(),
            context_length: config.context_length.unwrap_or(model_info.context_length),
            max_context_length: model_info.context_length,
            display_name,
            format: "gguf".to_string(),
            quantization,
            params_string,
            size_bytes,
            model_type: "llm".to_string(),
            vision: false,
            trained_for_tool_use: false,
            n_vocab: model_info.n_vocab,
            n_layer: model_info.n_layer,
            n_embd: model_info.n_embd,
            n_head: model_info.n_head,
        };

        let serving_engine =
            ServingEngine::new(&ctx, &model).map_err(|e| anyhow::anyhow!("Failed to create engine: {}", e))?;

        let loaded = Arc::new(LoadedModel {
            identifier: identifier.to_string(),
            tokenizer,
            serving_engine: Mutex::new(serving_engine),
            model,
            ctx,
            info: full_info.clone(),
            last_used: RwLock::new(Instant::now()),
            ttl: config.ttl,
            no_hup: config.no_hup,
        });

        self.models.insert(identifier.to_string(), loaded);
        tracing::info!(
            "Loaded model '{}': {} ({}L, {})",
            identifier,
            model_info.architecture,
            model_info.n_layer,
            full_info.quantization.as_deref().unwrap_or("unknown")
        );

        Ok(full_info)
    }

    /// Unload a model by identifier
    pub fn unload(&self, identifier: &str) -> anyhow::Result<()> {
        match self.models.remove(identifier) {
            Some(_) => {
                tracing::info!("Unloaded model '{}'", identifier);
                Ok(())
            }
            None => Err(anyhow::anyhow!("Model '{}' is not loaded", identifier)),
        }
    }

    /// Get a loaded model by identifier
    pub fn get(&self, identifier: &str) -> Option<Arc<LoadedModel>> {
        self.models.get(identifier).map(|r| r.clone())
    }

    /// List all loaded models
    pub fn list_loaded(&self) -> Vec<FullModelInfo> {
        self.models
            .iter()
            .map(|entry| entry.value().info.clone())
            .collect()
    }

    /// List all downloaded models (not necessarily loaded)
    pub fn list_downloaded(&self) -> Vec<DownloadedModelInfo> {
        let mut models = Vec::new();

        // Scan local models directory
        if let Ok(local) = scan_models_dir(self.resolver.models_dir()) {
            for entry in local {
                models.push(DownloadedModelInfo {
                    path: entry.path.to_string_lossy().to_string(),
                    name: entry.name,
                    size_bytes: entry.size_bytes,
                    quantization: entry.quantization,
                });
            }
        }

        // Scan HuggingFace cache
        if let Ok(hf) = scan_hf_cache(self.resolver.hf_cache_dir()) {
            for entry in hf {
                models.push(DownloadedModelInfo {
                    path: entry.path.to_string_lossy().to_string(),
                    name: entry.name,
                    size_bytes: entry.size_bytes,
                    quantization: entry.quantization,
                });
            }
        }

        models
    }

    /// Check for models that should be auto-unloaded due to TTL
    pub fn check_ttl_unload(&self) -> Vec<String> {
        let now = Instant::now();
        let mut to_unload = Vec::new();

        for entry in self.models.iter() {
            let model = entry.value();
            if model.no_hup {
                continue;
            }
            if let Some(ttl) = model.ttl {
                let last_used = *model.last_used.blocking_read();
                if now.duration_since(last_used) > ttl {
                    to_unload.push(model.identifier.clone());
                }
            }
        }

        for id in &to_unload {
            let _ = self.unload(id);
        }

        to_unload
    }

}

/// Extract quantization from filename (e.g., "Q4_K_M" from "model-Q4_K_M.gguf")
fn extract_quantization(path: &Path) -> Option<String> {
    let filename = path.file_name()?.to_str()?;

    // Common quantization patterns
    let patterns = [
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
        "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "F16", "F32",
        "BF16",
    ];

    for pattern in patterns {
        if filename.contains(pattern) {
            return Some(pattern.to_string());
        }
    }

    None
}

/// Estimate parameter count from architecture info
fn estimate_params(arch: &str, n_layer: usize, n_embd: usize) -> String {
    // Rough estimation based on common architectures
    let params = match arch.to_lowercase().as_str() {
        "llama" | "mistral" | "gemma" => {
            // Approximate: 4 * n_embd^2 * n_layer (for attention + FFN)
            let estimate = 4.0 * (n_embd as f64).powi(2) * (n_layer as f64);
            estimate / 1e9 // Convert to billions
        }
        _ => {
            // Generic estimate
            let estimate = 4.0 * (n_embd as f64).powi(2) * (n_layer as f64);
            estimate / 1e9
        }
    };

    if params >= 1.0 {
        format!("{:.1}B", params)
    } else {
        format!("{:.0}M", params * 1000.0)
    }
}
