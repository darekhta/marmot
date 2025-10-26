//! `marmot-lm ps` command - list currently loaded models on server

use crate::client::WsClient;

const DEFAULT_PORT: u16 = 1234;

pub async fn run() -> anyhow::Result<()> {
    // Connect to server
    let client = match WsClient::connect(DEFAULT_PORT, "llm").await {
        Ok(c) => c,
        Err(_) => {
            println!("No server running on port {}", DEFAULT_PORT);
            return Ok(());
        }
    };

    let models = client.llm().list_loaded().await?;

    if models.is_empty() {
        println!("No models currently loaded");
    } else {
        println!("{:<50} {:>10}", "NAME", "STATUS");
        for model in models {
            println!("{:<50} {:>10}", model.identifier, "loaded");
        }
    }

    client.close().await?;
    Ok(())
}
