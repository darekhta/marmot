//! `marmot-lm list` command - list available models

use crate::config::{scan_hf_cache, scan_models_dir, ModelResolver};

pub async fn run() -> anyhow::Result<()> {
    let resolver = ModelResolver::new();
    let models_dir = resolver.models_dir();
    let hf_cache = resolver.hf_cache_dir();

    let local_models = scan_models_dir(models_dir)?;
    let hf_models = scan_hf_cache(hf_cache)?;
    let total = local_models.len() + hf_models.len();

    if total == 0 {
        print_empty_state();
        return Ok(());
    }

    // Print header
    println!("{:<44} {:>8} {:>10}", "NAME", "QUANT", "SIZE");

    // Print all models
    for model in local_models.iter().chain(hf_models.iter()) {
        let quant = model.quantization.as_deref().unwrap_or("-");
        println!(
            "{:<44} {:>8} {:>9.1} GB",
            truncate(&model.name, 44),
            quant,
            model.size_gb()
        );
    }

    Ok(())
}

fn print_empty_state() {
    println!("No models found.");
    println!();
    println!("To get started, download a model:");
    println!();
    println!("  marmot-lm pull bartowski/Llama-3.2-1B-Instruct-GGUF");
    println!();
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
