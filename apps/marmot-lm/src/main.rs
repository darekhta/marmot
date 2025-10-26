//! marmot-lm: LLM inference server and CLI powered by marmot

mod ffi;

mod cli;
mod client;
mod common;
mod config;
mod embedded;
mod engine;
mod protocol;
mod server;
mod template;

use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "marmot-lm")]
#[command(about = "LLM inference server and CLI powered by marmot")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the WebSocket server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "1234")]
        port: u16,
        /// Run server in background (daemon mode)
        #[arg(short, long)]
        daemon: bool,
    },
    /// Run inference (uses server if available, otherwise embedded)
    Run {
        /// Model identifier or path
        model: String,
        /// Initial prompt
        #[arg(short, long)]
        prompt: Option<String>,
        /// Sampling temperature (0 = greedy/fastest, default 0.7)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
    },
    /// Download a model from HuggingFace
    Pull {
        /// Model identifier (e.g., "bartowski/Llama-3.2-1B-Instruct-GGUF")
        model: String,
        /// Preferred quantization (e.g., "Q4_K_M", "Q8_0")
        #[arg(short, long)]
        quantization: Option<String>,
    },
    /// List available models
    List,
    /// Show running models on server
    Ps,
    /// Stop the running server
    Stop,
    /// Show model information
    Info {
        /// Model path
        model: String,
    },
    /// Unload a model from server memory
    Unload {
        /// Model identifier
        model: String,
    },
    /// Remove a local model
    Rm {
        /// Model name or identifier
        model: String,
        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
}

fn init_logging() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "marmot_lm=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, daemon } => {
            if !daemon {
                tracing::info!("Starting server on port {}", port);
            }
            cli::serve::run(port, daemon).await
        }
        Commands::Run { model, prompt, temperature } => cli::run::run(&model, prompt.as_deref(), temperature).await,
        Commands::Pull {
            model,
            quantization,
        } => cli::pull::run(&model, quantization.as_deref()).await,
        Commands::List => cli::list::run().await,
        Commands::Ps => cli::ps::run().await,
        Commands::Stop => cli::stop::run().await,
        Commands::Info { model } => cli::info::run(&model).await,
        Commands::Unload { model } => cli::unload::run(&model).await,
        Commands::Rm { model, force } => cli::rm::run(&model, force).await,
    }
}
