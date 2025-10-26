//! `marmot-lm rm` command - remove a local model

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::config::{scan_hf_cache, scan_models_dir, ModelResolver};

pub async fn run(model: &str, force: bool) -> anyhow::Result<()> {
    let resolver = ModelResolver::new();

    // Find the model
    let (path, is_hf_cache) = find_model_to_remove(&resolver, model)?;

    let size_bytes = if is_hf_cache {
        // For HF cache, we remove the entire model directory
        calculate_dir_size(&path)?
    } else {
        path.metadata()?.len()
    };

    let size_str = format_size(size_bytes);

    // Confirm deletion
    if !force {
        print!(
            "Delete '{}' ({})? This cannot be undone. [y/N] ",
            path.display(),
            size_str
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled");
            return Ok(());
        }
    }

    // Remove the file or directory
    if is_hf_cache {
        fs::remove_dir_all(&path)?;
    } else {
        fs::remove_file(&path)?;
    }

    println!("Deleted '{}'", path.display());
    Ok(())
}

fn find_model_to_remove(resolver: &ModelResolver, specifier: &str) -> anyhow::Result<(PathBuf, bool)> {
    let models_dir = resolver.models_dir();
    let hf_cache_dir = resolver.hf_cache_dir();

    // 1. Check local models directory for exact match
    let local_path = models_dir.join(format!("{}.gguf", specifier));
    if local_path.exists() {
        return Ok((local_path, false));
    }

    // 2. Search local models directory
    let local_models = scan_models_dir(models_dir)?;
    let specifier_lower = specifier.to_lowercase();

    for model in &local_models {
        if model.name.to_lowercase() == specifier_lower
            || model.name.to_lowercase().contains(&specifier_lower)
        {
            return Ok((model.path.clone(), false));
        }
    }

    // 3. Check HuggingFace cache
    // Convert specifier to cache directory name
    let cache_name = format!("models--{}", specifier.replace('/', "--"));
    let cache_path = hf_cache_dir.join(&cache_name);

    if cache_path.exists() {
        return Ok((cache_path, true));
    }

    // 4. Search HF cache for partial match
    let hf_models = scan_hf_cache(hf_cache_dir)?;

    for model in &hf_models {
        if model.name.to_lowercase() == specifier_lower
            || model.name.to_lowercase().contains(&specifier_lower)
        {
            // For HF models, we need to find the parent cache directory
            // path is: .../models--org--repo/snapshots/hash/file.gguf
            // We want: .../models--org--repo
            if let Some(hash_dir) = model.path.parent() {
                if let Some(snapshots_dir) = hash_dir.parent() {
                    if let Some(cache_dir) = snapshots_dir.parent() {
                        return Ok((cache_dir.to_path_buf(), true));
                    }
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "Model '{}' not found.\n\nRun 'marmot-lm list' to see available models.",
        specifier
    ))
}

fn calculate_dir_size(path: &PathBuf) -> anyhow::Result<u64> {
    let mut total = 0u64;

    if path.is_file() {
        return Ok(path.metadata()?.len());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            total += path.metadata()?.len();
        } else if path.is_dir() {
            total += calculate_dir_size(&path)?;
        }
    }

    Ok(total)
}

fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
