//! `marmot-lm stop` command - stop the running server

use std::fs;
use std::time::Duration;

use crate::cli::serve::get_pid_file;

pub async fn run() -> anyhow::Result<()> {
    let pid_file = get_pid_file()?;

    if !pid_file.exists() {
        println!("No server is running (PID file not found)");
        return Ok(());
    }

    let content = fs::read_to_string(&pid_file)?;
    let pid: u32 = content.trim().parse()?;

    println!("Stopping server with PID {}...", pid);

    // Send SIGTERM to the process
    #[cfg(unix)]
    {
        use std::process::Command;

        let output = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .output()?;

        if !output.status.success() {
            // Process might already be dead
            if pid_file.exists() {
                fs::remove_file(&pid_file)?;
            }
            println!("Server was not running");
            return Ok(());
        }

        // Wait for the server to stop (up to 5 seconds)
        for i in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Check if process is still running
            let check = Command::new("kill")
                .args(["-0", &pid.to_string()])
                .output()?;

            if !check.status.success() {
                // Process has stopped
                if pid_file.exists() {
                    fs::remove_file(&pid_file)?;
                }
                println!("Server stopped");
                return Ok(());
            }

            if i == 10 {
                println!("Waiting for server to finish active connections...");
            }
        }

        // Force kill if still running
        println!("Server did not stop gracefully, forcing...");
        Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .output()?;

        if pid_file.exists() {
            fs::remove_file(&pid_file)?;
        }
        println!("Server killed");
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("Stop command is only supported on Unix systems");
    }

    Ok(())
}
