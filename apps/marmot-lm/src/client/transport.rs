use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use uuid::Uuid;

#[derive(serde::Deserialize)]
struct GreetingResponse {
    lmstudio: bool,
}

pub struct WsTransport {
    tx: mpsc::Sender<Message>,
}

impl WsTransport {
    pub async fn connect(
        port: u16,
        namespace: &str,
    ) -> anyhow::Result<(Self, mpsc::Receiver<Value>)> {
        let greeting_url = format!("http://localhost:{}/lmstudio-greeting", port);
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .map_err(|err| anyhow::anyhow!("Failed to build HTTP client: {}", err))?;
        let resp = client.get(&greeting_url).send().await?;
        let greeting: GreetingResponse = resp.json().await?;

        if !greeting.lmstudio {
            anyhow::bail!("Not an LM Studio compatible server");
        }

        let ws_url = format!("ws://localhost:{}/{}", port, namespace);
        let (ws_stream, _) = connect_async(&ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        let client_identifier = format!("guest:{}", Uuid::new_v4());
        let client_passkey = Uuid::new_v4().to_string();
        let auth_msg = json!({
            "authVersion": 1,
            "clientIdentifier": client_identifier,
            "clientPasskey": client_passkey,
        });
        write
            .send(Message::Text(serde_json::to_string(&auth_msg)?.into()))
            .await?;

        match read.next().await {
            Some(Ok(Message::Text(text))) => {
                let value: Value = serde_json::from_str(&text)?;
                if value.get("success").and_then(Value::as_bool) != Some(true) {
                    let error = value
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("Authentication failed");
                    anyhow::bail!("Authentication failed: {}", error);
                }
            }
            Some(Ok(Message::Close(_))) | None => {
                anyhow::bail!("Connection closed during authentication");
            }
            Some(Ok(_)) => {
                anyhow::bail!("Unexpected authentication response");
            }
            Some(Err(e)) => {
                anyhow::bail!("Authentication failed: {}", e);
            }
        }

        let (tx, mut rx) = mpsc::channel::<Message>(32);
        let (incoming_tx, incoming_rx) = mpsc::channel::<Value>(64);

        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if write.send(msg).await.is_err() {
                    break;
                }
            }
        });

        tokio::spawn(async move {
            while let Some(Ok(msg)) = read.next().await {
                if let Message::Text(text) = msg {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&text) {
                        let _ = incoming_tx.send(parsed).await;
                    }
                }
            }
        });

        Ok((Self { tx }, incoming_rx))
    }

    pub async fn send(&self, msg: &Value) -> anyhow::Result<()> {
        let text = serde_json::to_string(msg)?;
        self.tx.send(Message::Text(text.into())).await?;
        Ok(())
    }

    pub async fn close(&self) -> anyhow::Result<()> {
        self.tx.send(Message::Close(None)).await?;
        Ok(())
    }
}
