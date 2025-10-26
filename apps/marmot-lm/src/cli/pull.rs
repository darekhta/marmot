//! `marmot-lm pull` command - download models from HuggingFace

use hf_hub::api::sync::ApiBuilder;

pub async fn run(model_id: &str, quantization: Option<&str>) -> anyhow::Result<()> {
    println!("Fetching {}...", model_id);

    // Build API with progress bar enabled
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(model_id.to_string());

    // List files in the repo
    let info = repo.info()?;

    // Collect GGUF files
    let gguf_files: Vec<String> = info
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .map(|s| s.rfilename.clone())
        .collect();

    if gguf_files.is_empty() {
        anyhow::bail!("No GGUF files found in repository {}", model_id);
    }

    // Select file based on quantization preference
    let file_to_download = select_gguf_file(&gguf_files, quantization)?;

    println!("Downloading {}...", file_to_download);

    // Download with hf-hub's built-in progress bar
    let path = repo.get(file_to_download)?;

    println!();
    println!("Done! Run inference with:");
    println!();
    println!("  marmot-lm run {}", path.display());
    println!();

    Ok(())
}

fn select_gguf_file<'a>(files: &'a [String], preference: Option<&str>) -> anyhow::Result<&'a str> {
    // Priority order for quantization
    let quant_priority = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0", "Q3_K_M",
    ];

    if let Some(pref) = preference {
        // User specified a preference
        if let Some(file) = files.iter().find(|f| f.contains(pref)) {
            return Ok(file);
        }
        println!(
            "Warning: No file matching '{}' found, using default selection",
            pref
        );
    }

    // Try to find the best quantization in order of preference
    for quant in quant_priority {
        if let Some(file) = files.iter().find(|f| f.contains(quant)) {
            return Ok(file);
        }
    }

    // Fall back to first file
    files
        .first()
        .map(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("No suitable GGUF file found"))
}
