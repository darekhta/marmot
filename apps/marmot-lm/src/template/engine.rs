//! Template engine implementation.

use minijinja::Environment;
use thiserror::Error;

use crate::common::SimpleMessage;

use super::context::RenderContext;
use super::filters;
use super::source;
use super::stop_strings::StopStringResolver;
use super::RenderResult;

/// Errors that can occur during template rendering.
#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("Template compilation failed: {0}")]
    Compilation(String),

    #[error("Template rendering failed: {0}")]
    Render(String),
}

impl From<minijinja::Error> for TemplateError {
    fn from(err: minijinja::Error) -> Self {
        TemplateError::Render(err.to_string())
    }
}

/// Chat template engine using Jinja2 (minijinja).
pub struct TemplateEngine {
    env: Environment<'static>,
    architecture: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl TemplateEngine {
    /// Create a new template engine.
    ///
    /// # Arguments
    /// * `gguf_template` - Template from GGUF metadata, if available
    /// * `architecture` - Model architecture name for fallback selection
    /// * `model_name` - Model name for detecting instruct vs base models
    /// * `bos_token` - BOS token string
    /// * `eos_token` - EOS token string
    pub fn new(
        gguf_template: Option<&str>,
        architecture: &str,
        model_name: Option<&str>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self, TemplateError> {
        let source = source::select_source(gguf_template, architecture, model_name);

        let mut env = Environment::new();
        filters::register_filters(&mut env);

        // Get the template source
        let template_source = source.template().to_string();

        // Use loader for dynamic templates
        env.set_loader(move |name| {
            if name == "chat" {
                Ok(Some(template_source.clone()))
            } else {
                Ok(None)
            }
        });

        // Validate the template compiles
        env.get_template("chat")
            .map_err(|e| TemplateError::Compilation(e.to_string()))?;

        Ok(Self {
            env,
            architecture: architecture.to_string(),
            bos_token,
            eos_token,
        })
    }

    /// Render messages to a formatted prompt.
    pub fn render(
        &self,
        messages: &[SimpleMessage],
        add_generation_prompt: bool,
    ) -> Result<RenderResult, TemplateError> {
        let ctx = RenderContext::new(
            messages,
            add_generation_prompt,
            self.bos_token.clone(),
            self.eos_token.clone(),
        );

        let template = self.env.get_template("chat")?;
        let template_source = template.source();
        let raw_prompt = template.render(ctx.to_value())?;

        // Normalize whitespace: collapse multiple consecutive newlines to single,
        // and remove newlines immediately after </s> (Zephyr-style format fix)
        let mut prompt = String::with_capacity(raw_prompt.len());
        let mut prev_was_newline = false;
        for ch in raw_prompt.trim().chars() {
            if ch == '\n' {
                // Skip newlines immediately after </s>
                if prompt.ends_with("</s>") {
                    continue;
                }
                if !prev_was_newline {
                    prompt.push(ch);
                }
                prev_was_newline = true;
            } else {
                prompt.push(ch);
                prev_was_newline = false;
            }
        }

        // Ensure trailing newline after generation prompt (matches hardcoded format)
        if !prompt.is_empty() && !prompt.ends_with('\n') {
            prompt.push('\n');
        }

        // Get stop strings
        let mut stop_strings = StopStringResolver::resolve(
            Some(template_source),
            &self.architecture,
        );
        if template_source.contains("eos_token") {
            if let Some(eos_token) = self.eos_token.as_ref() {
                if !eos_token.is_empty() && !stop_strings.iter().any(|s| s == eos_token) {
                    stop_strings.push(eos_token.clone());
                }
            }
        }

        Ok(RenderResult {
            prompt,
            stop_strings,
        })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages() -> Vec<SimpleMessage> {
        vec![
            SimpleMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            },
            SimpleMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
            SimpleMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
            },
        ]
    }

    #[test]
    fn test_engine_with_gguf_template() {
        let template = r#"
{%- for message in messages %}
{{ message.role }}: {{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}
assistant:
{%- endif %}
"#;

        let engine = TemplateEngine::new(Some(template), "unknown", None, None, None).unwrap();
        let result = engine.render(&make_messages(), true).unwrap();
        assert!(result.prompt.contains("user: Hello!"));
        assert!(result.prompt.contains("assistant: Hi there!"));
        assert!(result.prompt.contains("assistant:"));
    }

    #[test]
    fn test_engine_with_fallback() {
        // Use "instruct" in model name to trigger architecture fallback
        let engine = TemplateEngine::new(None, "llama", Some("llama-instruct"), None, None).unwrap();
        let result = engine.render(&make_messages(), true).unwrap();
        assert!(result.prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.prompt.contains("<|eot_id|>"));
        assert!(result.stop_strings.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_engine_chatml() {
        let engine = TemplateEngine::new(None, "qwen2", Some("Qwen2-Instruct"), None, None).unwrap();

        let result = engine.render(&make_messages(), true).unwrap();
        assert!(result.prompt.contains("<|im_start|>user"));
        assert!(result.prompt.contains("<|im_end|>"));
        assert!(result.stop_strings.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_engine_gemma() {
        let engine = TemplateEngine::new(None, "gemma", Some("gemma-2b-it"), None, None).unwrap();

        let result = engine.render(&make_messages(), true).unwrap();
        assert!(result.prompt.contains("<start_of_turn>user"));
        assert!(result.prompt.contains("<start_of_turn>model"));
        assert!(result.stop_strings.contains(&"<end_of_turn>".to_string()));
    }

    #[test]
    fn test_engine_with_bos_token() {
        let engine =
            TemplateEngine::new(None, "llama", Some("llama-chat"), Some("<|begin_of_text|>".to_string()), None)
                .unwrap();

        let result = engine.render(&make_messages(), false).unwrap();
        assert!(result.prompt.starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn test_all_architecture_templates() {
        // Test all supported architectures from the fixtures
        // Note: mistral maps to llama3 format (for modern Mistral v0.2+ models)
        // Each test uses "-instruct" suffix to trigger architecture fallback
        let architectures = [
            ("llama", "<|start_header_id|>", "<|eot_id|}"),
            ("llama3", "<|start_header_id|>", "<|eot_id|>"),
            ("gemma", "<start_of_turn>", "<end_of_turn>"),
            ("gemma2", "<start_of_turn>", "<end_of_turn>"),
            ("qwen", "<|im_start|>", "<|im_end|>"),
            ("qwen2", "<|im_start|>", "<|im_end|>"),
            ("phi", "<|user|>", "<|end|>"),
            ("phi3", "<|user|>", "<|end|>"),
            ("mistral", "<|start_header_id|>", "<|eot_id|>"), // Modern Mistral uses llama3 format
            ("yi", "<|im_start|>", "<|im_end|}"),             // Yi uses ChatML
            ("deepseek", "<|im_start|>", "<|im_end|>"),       // DeepSeek uses ChatML
        ];

        let messages = make_messages();

        for (arch, expected_marker, _expected_stop) in architectures {
            let model_name = format!("{}-instruct", arch);
            let engine = TemplateEngine::new(None, arch, Some(&model_name), None, None)
                .unwrap_or_else(|e| panic!("Failed to create engine for {}: {}", arch, e));

            let result = engine.render(&messages, true)
                .unwrap_or_else(|e| panic!("Failed to render for {}: {}", arch, e));

            assert!(
                result.prompt.contains(expected_marker),
                "Architecture '{}' should contain '{}' in prompt.\nGot:\n{}",
                arch,
                expected_marker,
                result.prompt
            );
        }
    }

    #[test]
    fn test_phi3_template_format() {
        let engine = TemplateEngine::new(None, "phi3", Some("phi3-instruct"), None, None).unwrap();
        let result = engine.render(&make_messages(), true).unwrap();

        assert!(result.prompt.contains("<|user|>"));
        assert!(result.prompt.contains("<|assistant|>"));
        assert!(result.prompt.contains("<|end|>"));
        assert!(result.stop_strings.contains(&"<|end|>".to_string()));
    }

    #[test]
    fn test_llama2_uses_mistral_format() {
        // llama2 uses the older [INST] format
        let engine = TemplateEngine::new(None, "llama2", Some("llama2-chat"), None, None).unwrap();
        let result = engine.render(&make_messages(), true).unwrap();

        assert!(result.prompt.contains("[INST]"));
        assert!(result.prompt.contains("[/INST]"));
        assert!(result.stop_strings.contains(&"[/INST]".to_string()));
    }

    /// Visual test to show template output for all architectures.
    /// Run with: cargo test visual_template_output -- --nocapture --ignored
    #[test]
    #[ignore]
    fn visual_template_output() {
        let messages = make_messages();
        let architectures = [
            "llama",
            "llama2",
            "gemma",
            "qwen2",
            "qwen3",
            "phi3",
            "yi",
            "deepseek",
        ];

        for arch in architectures {
            println!("\n{}", "=".repeat(60));
            println!("Architecture: {}", arch);
            println!("{}", "=".repeat(60));

            let model_name = format!("{}-instruct", arch);
            let engine = TemplateEngine::new(None, arch, Some(&model_name), None, None).unwrap();
            let result = engine.render(&messages, true).unwrap();

            println!("Prompt:\n{}", result.prompt);
            println!("\nStop strings: {:?}", result.stop_strings);
        }
    }
}
