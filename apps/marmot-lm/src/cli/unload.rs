//! `marmot-lm unload` command - unload a model from server memory

use crate::client::WsClient;

const DEFAULT_PORT: u16 = 1234;

pub async fn run(model: &str) -> anyhow::Result<()> {
    // Connect to server
    let client = match WsClient::connect(DEFAULT_PORT, "llm").await {
        Ok(c) => c,
        Err(_) => {
            anyhow::bail!(
                "Could not connect to server on port {}. Is the server running?",
                DEFAULT_PORT
            );
        }
    };

    // Call unloadModel RPC
    let llm = client.llm();
    match llm.unload_model(model).await {
        Ok(_) => {
            println!("Unloaded model '{}'", model);
        }
        Err(e) => {
            // Check if it's a "not loaded" error
            let err_str = e.to_string();
            if err_str.contains("not loaded") {
                println!("Model '{}' is not loaded", model);
            } else {
                anyhow::bail!("Failed to unload model: {}", e);
            }
        }
    }

    client.close().await?;
    Ok(())
}
