//! In-process inference runtime
//!
//! This module provides embedded (no-server) inference capability.
//! It uses libmarmot for all model operations - no duplicate logic.

use crate::common::SimpleMessage;
use crate::ffi::{Backend, Context, GenerateOptions, Model, SamplingOptions, ServingEngine, Tokenizer};
use crate::server::rpc::format_messages_with_defaults;
use std::io::{self, BufRead, Write};

/// Embedded runtime - thin wrapper around libmarmot
pub struct EmbeddedRuntime {
    tokenizer: Tokenizer,
    /// Chat template from GGUF metadata (if present)
    chat_template: Option<String>,
    /// Model architecture for fallback template selection
    architecture: String,
    /// Model name for detecting instruct vs base models
    model_name: Option<String>,
    /// Persistent engine - reused across generation calls to avoid memory churn
    engine: Option<ServingEngine>,
    // Kept alive to maintain device resources
    model: Model,
    ctx: Context,
}

impl EmbeddedRuntime {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let backend = match std::env::var("MARMOT_BACKEND").ok().as_deref() {
            Some("cpu") | Some("CPU") => Backend::Cpu,
            Some("metal") | Some("METAL") => Backend::Metal,
            _ => Backend::Metal,
        };
        let ctx = Context::new(backend).map_err(|e| anyhow::anyhow!("{}", e))?;
        let model = Model::load(model_path).map_err(|e| anyhow::anyhow!("{}", e))?;
        let tokenizer = Tokenizer::from_gguf(model_path).map_err(|e| anyhow::anyhow!("{}", e))?;

        // Get model info for template selection
        let model_info = model.info().map_err(|e| anyhow::anyhow!("{}", e))?;
        let architecture = model_info.architecture.clone();
        // Extract model name from path (e.g., "gemma-2b.Q4_K_M.gguf" -> "gemma-2b.Q4_K_M")
        let model_name = std::path::Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string());

        // Get chat template from GGUF metadata (libmarmot handles this)
        let chat_template = tokenizer.chat_template().ok().flatten();

        if chat_template.is_some() {
            tracing::info!("Model has chat template");
        } else {
            tracing::info!(
                architecture = %architecture,
                model_name = ?model_name,
                "No chat template in model, will detect instruct vs base from model name"
            );
        }

        Ok(Self {
            tokenizer,
            chat_template,
            architecture,
            model_name,
            engine: None,
            model,
            ctx,
        })
    }

    pub fn model_info(&self) -> anyhow::Result<crate::ffi::ModelInfo> {
        self.model.info().map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Generate response from a prompt string (applies chat template when available).
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingOptions,
    ) -> anyhow::Result<String> {
        let (formatted, stop_strings) = self.format_prompt(prompt);
        self.generate_raw(&formatted, max_tokens, sampling, &stop_strings, true)
    }

    fn format_prompt(&self, prompt: &str) -> (String, Vec<String>) {
        let messages = [SimpleMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        format_messages_with_defaults(
            &messages,
            self.chat_template.as_deref(),
            &self.architecture,
            self.model_name.as_deref(),
            self.tokenizer.bos_token_string(),
            self.tokenizer.eos_token_string(),
            true,
        )
    }

    fn stop_tokens_for_strings(&self, stop_strings: &[String]) -> Vec<i32> {
        let mut tokens = Vec::new();
        for stop in stop_strings {
            if let Ok(Some(id)) = self.tokenizer.piece_to_token(stop) {
                tokens.push(id);
            }
        }
        // Also add EOS token if we have stop strings that look like end markers
        if let Ok(special) = self.tokenizer.special_tokens() {
            if let Some(eos_id) = special.eos_id {
                if !tokens.contains(&eos_id) {
                    tokens.push(eos_id);
                }
            }
        }
        tokens.sort_unstable();
        tokens.dedup();
        tokens
    }

    fn generate_raw(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingOptions,
        stop_strings: &[String],
        stream_output: bool,
    ) -> anyhow::Result<String> {
        let tokens = self
            .tokenizer
            .encode(prompt, true, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut max_new_tokens = max_tokens;
        if max_new_tokens == 0 {
            let info = self.model_info()?;
            max_new_tokens = info.context_length.saturating_sub(tokens.len()).max(1);
        }

        let stop_tokens = self.stop_tokens_for_strings(stop_strings);
        let gen_opts = GenerateOptions {
            max_new_tokens,
            stop_on_eos: true,
            stop_tokens: stop_tokens.clone(),
        };

        // Create or reuse engine - take it out to avoid borrow conflicts with tokenizer
        let mut engine = match self.engine.take() {
            Some(e) => e,
            None => ServingEngine::new(&self.ctx, &self.model).map_err(|e| anyhow::anyhow!("{}", e))?,
        };

        let request_id = engine
            .submit(&tokens, &gen_opts, sampling)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut generated = String::new();
        let mut sent_tokens = 0;
        let mut stdout = io::stdout();
        let mut needs_release = true;

        loop {
            engine.step(1).map_err(|e| anyhow::anyhow!("{}", e))?;

            match engine.request_state(request_id) {
                crate::ffi::RequestState::Prefill | crate::ffi::RequestState::Decoding => {
                    if let Ok(all_tokens) = engine.get_tokens(request_id) {
                        if all_tokens.len() > sent_tokens {
                            let new_tokens = &all_tokens[sent_tokens..];

                            // Check for stop tokens in raw tokens (before decode)
                            if new_tokens.iter().any(|t| stop_tokens.contains(t)) {
                                let _ = engine.cancel(request_id);
                                needs_release = false; // cancel removes request
                                break;
                            }

                            let text = self
                                .tokenizer
                                .decode_streaming(new_tokens)
                                .map_err(|e| anyhow::anyhow!("{}", e))?;
                            if stream_output && !text.is_empty() {
                                print!("{}", text);
                                stdout.flush()?;
                            }
                            generated.push_str(&text);
                            sent_tokens = all_tokens.len();

                            // String-based stop detection for multi-token sequences
                            if !stop_strings.is_empty()
                                && stop_strings.iter().any(|s| generated.contains(s))
                            {
                                let _ = engine.cancel(request_id);
                                needs_release = false; // cancel removes request
                                break;
                            }
                        }
                    }
                }
                crate::ffi::RequestState::Done => {
                    break;
                }
                crate::ffi::RequestState::Canceled => {
                    needs_release = false; // already canceled/removed
                    break;
                }
                crate::ffi::RequestState::Failed => {
                    let _ = engine.release(request_id);
                    self.engine = Some(engine); // Put back before returning
                    return Err(anyhow::anyhow!("Generation failed"));
                }
                crate::ffi::RequestState::Invalid => {
                    // Request was removed (e.g., after cancel) - treat as done
                    needs_release = false;
                    break;
                }
                _ => {}
            }
        }

        if needs_release {
            engine.release(request_id).map_err(|e| anyhow::anyhow!("{}", e))?;
        }
        self.engine = Some(engine); // Put engine back for reuse
        Ok(generated.trim().to_string())
    }

    /// Run interactive chat loop
    pub fn run_interactive(&mut self, system_prompt: Option<&str>) -> anyhow::Result<()> {
        let info = self.model_info()?;
        println!("Model: {} ({} layers)", info.architecture, info.n_layer);

        if self.chat_template.is_some() {
            println!("Chat template: available");
        } else {
            println!("Chat template: none (raw mode)");
        }
        println!("Type 'exit' or Ctrl+D to quit\n");

        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut messages: Vec<SimpleMessage> = Vec::new();

        // Add system prompt if provided
        if let Some(system) = system_prompt {
            messages.push(SimpleMessage {
                role: "system".to_string(),
                content: system.to_string(),
            });
        }

        loop {
            print!("> ");
            stdout.flush()?;

            let mut input = String::new();
            if stdin.lock().read_line(&mut input)? == 0 {
                println!();
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }
            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                break;
            }

            messages.push(SimpleMessage {
                role: "user".to_string(),
                content: input.to_string(),
            });

            let (prompt, stop_strings) = format_messages_with_defaults(
                &messages,
                self.chat_template.as_deref(),
                &self.architecture,
                self.model_name.as_deref(),
                self.tokenizer.bos_token_string(),
                self.tokenizer.eos_token_string(),
                true,
            );

            let sampling = SamplingOptions {
                suppress_special_tokens: true,
                ..Default::default()
            };
            match self.generate_raw(&prompt, 0, &sampling, &stop_strings, true) {
                Ok(response) => {
                    println!();
                    messages.push(SimpleMessage {
                        role: "assistant".to_string(),
                        content: response,
                    });
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        }

        Ok(())
    }
}
