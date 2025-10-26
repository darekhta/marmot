//! `marmot-lm serve` command - start WebSocket server

use std::fs;
use std::path::PathBuf;

use crate::server::start_server_with_shutdown;
use tokio_util::sync::CancellationToken;

/// Get the runtime directory for marmot-lm
pub fn get_runtime_dir() -> anyhow::Result<PathBuf> {
    let dir = dirs::runtime_dir()
        .or_else(|| dirs::cache_dir().map(|d| d.join("marmot-lm")))
        .ok_or_else(|| anyhow::anyhow!("Could not determine runtime directory"))?;
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Get the PID file path
pub fn get_pid_file() -> anyhow::Result<PathBuf> {
    Ok(get_runtime_dir()?.join("marmot-lm.pid"))
}

/// Check if server is already running
fn is_server_running() -> anyhow::Result<Option<u32>> {
    let pid_file = get_pid_file()?;
    if !pid_file.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&pid_file)?;
    let pid: u32 = content.trim().parse()?;

    // Check if process is running
    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()?;
        if output.status.success() {
            return Ok(Some(pid));
        }
    }

    // Stale PID file, remove it
    fs::remove_file(&pid_file)?;
    Ok(None)
}

/// Write PID file
fn write_pid_file() -> anyhow::Result<()> {
    let pid_file = get_pid_file()?;
    fs::write(&pid_file, std::process::id().to_string())?;
    Ok(())
}

/// Remove PID file
fn remove_pid_file() -> anyhow::Result<()> {
    let pid_file = get_pid_file()?;
    if pid_file.exists() {
        fs::remove_file(&pid_file)?;
    }
    Ok(())
}

pub async fn run(port: u16, daemon: bool) -> anyhow::Result<()> {
    // Check if server is already running
    if let Some(pid) = is_server_running()? {
        anyhow::bail!("Server is already running with PID {}", pid);
    }

    if daemon {
        // Daemon mode - fork to background
        #[cfg(unix)]
        {
            use std::process::Command;

            // Get current executable path
            let exe = std::env::current_exe()?;

            // Spawn detached process
            let child = Command::new(&exe)
                .args(["serve", "--port", &port.to_string()])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()?;

            println!("Server started in background with PID {}", child.id());
            println!("Run `marmot-lm stop` to stop the server");
            return Ok(());
        }

        #[cfg(not(unix))]
        {
            anyhow::bail!("Daemon mode is only supported on Unix systems");
        }
    }

    // Write PID file
    write_pid_file()?;

    // Setup signal handlers for graceful shutdown
    let shutdown = CancellationToken::new();
    let shutdown_clone = shutdown.clone();

    #[cfg(unix)]
    {
        tokio::spawn(async move {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Failed to setup SIGTERM handler");

            let mut sigint =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                    .expect("Failed to setup SIGINT handler");

            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM");
                }
                _ = sigint.recv() => {
                    tracing::info!("Received SIGINT");
                }
            }

            shutdown_clone.cancel();
        });
    }

    #[cfg(not(unix))]
    {
        tokio::spawn(async move {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to setup Ctrl+C handler");
            tracing::info!("Received Ctrl+C");
            shutdown_clone.cancel();
        });
    }

    // Run server
    let result = start_server_with_shutdown(port, shutdown).await;

    // Cleanup
    remove_pid_file()?;

    result
}
