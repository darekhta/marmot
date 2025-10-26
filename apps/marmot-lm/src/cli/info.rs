//! `marmot-lm info` command - show model information

use crate::config::ModelResolver;
use crate::ffi::{Model, Tokenizer};

pub async fn run(model_spec: &str) -> anyhow::Result<()> {
    // Resolve model name to path
    let resolver = ModelResolver::new();
    let model_path = resolver.resolve(model_spec)?;

    tracing::info!("Loading model info from: {}", model_path.display());

    // Load model
    let model = Model::load(&model_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    let info = model.info().map_err(|e| anyhow::anyhow!("{}", e))?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&model_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    let special = tokenizer.special_tokens().ok();

    println!("Model Information");
    println!("=================");
    println!("Architecture:     {}", info.architecture);
    println!("Context Length:   {}", info.context_length);
    println!("Vocabulary Size:  {}", info.n_vocab);
    println!("Embedding Dim:    {}", info.n_embd);
    println!("Layers:           {}", info.n_layer);
    println!("Attention Heads:  {}", info.n_head);
    println!("KV Heads:         {}", info.n_head_kv);
    println!("Feed-Forward:     {}", info.ff_length);
    println!("RoPE Dimension:   {}", info.rope_dimension);
    println!("RoPE Freq Base:   {}", info.rope_freq_base);
    println!("RMS Norm Eps:     {}", info.rms_norm_eps);
    println!();
    println!("Tokenizer");
    println!("---------");
    println!("Vocab Size:       {}", tokenizer.vocab_size());
    if let Some(sp) = special {
        println!(
            "BOS Token:        {}",
            sp.bos_id.map_or("None".to_string(), |id| id.to_string())
        );
        println!(
            "EOS Token:        {}",
            sp.eos_id.map_or("None".to_string(), |id| id.to_string())
        );
        println!(
            "UNK Token:        {}",
            sp.unk_id.map_or("None".to_string(), |id| id.to_string())
        );
        println!(
            "PAD Token:        {}",
            sp.pad_id.map_or("None".to_string(), |id| id.to_string())
        );
    }

    Ok(())
}
