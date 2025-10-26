//! Model registry and path resolution
//!
//! Resolves model specifiers (names, paths, HuggingFace repos) to file paths.

use std::fs;
use std::path::{Path, PathBuf};

/// Model resolver for finding GGUF files
pub struct ModelResolver {
    /// Local models directory (~/.local/share/marmot/models)
    models_dir: PathBuf,
    /// HuggingFace cache directory (~/.cache/huggingface/hub)
    hf_cache_dir: PathBuf,
}

impl ModelResolver {
    pub fn new() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("marmot");

        // HuggingFace cache directory resolution:
        // 1. HF_HUB_CACHE environment variable (explicit cache path)
        // 2. HF_HOME environment variable + /hub
        // 3. ~/.cache/huggingface/hub (Python default, cross-platform)
        // 4. dirs::cache_dir()/huggingface/hub (macOS ~/Library/Caches/...)
        let hf_cache_dir = std::env::var("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|_| std::env::var("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
            .unwrap_or_else(|_| {
                // First try ~/.cache/huggingface/hub (Python/hf-hub default)
                let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
                let xdg_cache = home.join(".cache").join("huggingface").join("hub");
                if xdg_cache.exists() {
                    xdg_cache
                } else {
                    // Fall back to platform-specific cache dir
                    dirs::cache_dir()
                        .unwrap_or_else(|| PathBuf::from(".cache"))
                        .join("huggingface")
                        .join("hub")
                }
            });

        Self {
            models_dir: data_dir.join("models"),
            hf_cache_dir,
        }
    }

    /// Resolve a model specifier to a file path
    ///
    /// Supports:
    /// - Full path: /path/to/model.gguf
    /// - Local name: llama-3.2-1b (looks in models_dir)
    /// - HuggingFace repo: bartowski/Llama-3.2-1B-Instruct-GGUF
    pub fn resolve(&self, specifier: &str) -> anyhow::Result<PathBuf> {
        // 1. Check if it's a direct path that exists
        let path = Path::new(specifier);
        if path.exists() && path.is_file() {
            return Ok(path.to_path_buf());
        }

        // 2. Check if it's a path with .gguf extension
        if specifier.ends_with(".gguf") && path.exists() {
            return Ok(path.to_path_buf());
        }

        // 3. Check local models directory (with .gguf extension)
        let local_path = self.models_dir.join(format!("{}.gguf", specifier));
        if local_path.exists() {
            return Ok(local_path);
        }

        // 4. Check local models directory (exact match)
        let local_exact = self.models_dir.join(specifier);
        if local_exact.exists() && local_exact.is_file() {
            return Ok(local_exact);
        }

        // 5. Search local models directory for partial match
        if let Some(path) = self.find_in_models_dir(specifier)? {
            return Ok(path);
        }

        // 6. Check HuggingFace cache
        if let Some(path) = self.find_in_hf_cache(specifier)? {
            return Ok(path);
        }

        Err(anyhow::anyhow!(
            "Model not found: '{}'\n\
             Searched:\n  \
             - Direct path\n  \
             - {}\n  \
             - {}\n\n\
             Try:\n  \
             - marmot-lm pull <huggingface-repo>\n  \
             - Specify full path to .gguf file",
            specifier,
            self.models_dir.display(),
            self.hf_cache_dir.display()
        ))
    }

    /// Search local models directory for partial filename match
    fn find_in_models_dir(&self, specifier: &str) -> anyhow::Result<Option<PathBuf>> {
        if !self.models_dir.exists() {
            return Ok(None);
        }

        let specifier_lower = specifier.to_lowercase();

        for entry in fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            if let Some(ext) = path.extension() {
                if ext != "gguf" {
                    continue;
                }
            } else {
                continue;
            }

            // Check if filename contains the specifier
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let name_lower = name.to_lowercase();
                if name_lower.contains(&specifier_lower) {
                    return Ok(Some(path));
                }
            }
        }

        Ok(None)
    }

    /// Search HuggingFace cache for model
    ///
    /// HuggingFace cache structure:
    /// ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/*.gguf
    fn find_in_hf_cache(&self, specifier: &str) -> anyhow::Result<Option<PathBuf>> {
        if !self.hf_cache_dir.exists() {
            return Ok(None);
        }

        // Convert specifier to cache directory name
        // "bartowski/Llama-3.2-1B-Instruct-GGUF" -> "models--bartowski--Llama-3.2-1B-Instruct-GGUF"
        let cache_name = format!("models--{}", specifier.replace('/', "--"));
        let cache_path = self.hf_cache_dir.join(&cache_name);

        if cache_path.exists() {
            return self.find_gguf_in_cache_dir(&cache_path);
        }

        // Try partial match on cache directories
        let specifier_lower = specifier.to_lowercase();
        for entry in fs::read_dir(&self.hf_cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with("models--") {
                    continue;
                }

                let name_lower = name.to_lowercase();
                if name_lower.contains(&specifier_lower) {
                    if let Some(gguf) = self.find_gguf_in_cache_dir(&path)? {
                        return Ok(Some(gguf));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Find a GGUF file in a HuggingFace cache directory
    fn find_gguf_in_cache_dir(&self, cache_dir: &Path) -> anyhow::Result<Option<PathBuf>> {
        let snapshots_dir = cache_dir.join("snapshots");
        if !snapshots_dir.exists() {
            return Ok(None);
        }

        // Get all snapshots, sorted by modification time (newest first)
        let mut snapshots: Vec<_> = fs::read_dir(&snapshots_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();

        snapshots.sort_by(|a, b| {
            let a_time = a.metadata().and_then(|m| m.modified()).ok();
            let b_time = b.metadata().and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time) // Newest first
        });

        // Search for GGUF files in snapshots
        for snapshot in snapshots {
            // Prefer Q4_K_M quantization
            let mut gguf_files: Vec<PathBuf> = Vec::new();

            for entry in fs::read_dir(&snapshot)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "gguf" {
                            gguf_files.push(path);
                        }
                    }
                }
            }

            if gguf_files.is_empty() {
                continue;
            }

            // Prefer Q4_K_M, then Q4_K_S, then any
            let preferred = gguf_files
                .iter()
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.contains("Q4_K_M"))
                        .unwrap_or(false)
                })
                .or_else(|| {
                    gguf_files.iter().find(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.contains("Q4_K"))
                            .unwrap_or(false)
                    })
                })
                .or(gguf_files.first());

            if let Some(path) = preferred {
                return Ok(Some(path.clone()));
            }
        }

        Ok(None)
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Get the HuggingFace cache directory path
    pub fn hf_cache_dir(&self) -> &Path {
        &self.hf_cache_dir
    }
}

impl Default for ModelResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a local model
#[derive(Debug)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub quantization: Option<String>,
}

impl ModelEntry {
    pub fn from_path(path: PathBuf) -> anyhow::Result<Self> {
        let name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let size_bytes = path.metadata()?.len();

        // Try to extract quantization from filename
        let quantization = path.file_name().and_then(|n| n.to_str()).and_then(|name| {
            // Common patterns: Q4_K_M, Q5_K_S, Q8_0, etc.
            let patterns = [
                "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K", "Q2_K",
            ];
            for pattern in patterns {
                if name.contains(pattern) {
                    return Some(pattern.to_string());
                }
            }
            None
        });

        Ok(Self {
            name,
            path,
            size_bytes,
            quantization,
        })
    }

    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / 1_073_741_824.0
    }
}

/// Scan the local models directory for GGUF files
pub fn scan_models_dir(dir: &Path) -> anyhow::Result<Vec<ModelEntry>> {
    let mut models = Vec::new();

    if !dir.exists() {
        return Ok(models);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        if let Some(ext) = path.extension() {
            if ext == "gguf" {
                if let Ok(model) = ModelEntry::from_path(path) {
                    models.push(model);
                }
            }
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

/// Scan HuggingFace cache for GGUF models
pub fn scan_hf_cache(cache_dir: &Path) -> anyhow::Result<Vec<ModelEntry>> {
    let mut models = Vec::new();

    if !cache_dir.exists() {
        return Ok(models);
    }

    for entry in fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };

        if !dir_name.starts_with("models--") {
            continue;
        }

        // Look for GGUF files in snapshots
        let snapshots_dir = path.join("snapshots");
        if !snapshots_dir.exists() {
            continue;
        }

        for snapshot in fs::read_dir(&snapshots_dir)?.filter_map(|e| e.ok()) {
            let snapshot_path = snapshot.path();
            if !snapshot_path.is_dir() {
                continue;
            }

            for file in fs::read_dir(&snapshot_path)?.filter_map(|e| e.ok()) {
                let file_path = file.path();
                if file_path.is_file() {
                    if let Some(ext) = file_path.extension() {
                        if ext == "gguf" {
                            if let Ok(model) = ModelEntry::from_path(file_path) {
                                models.push(model);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}
