//! Chat message formatting using Jinja2 templates.

use crate::common::SimpleMessage;
use crate::template::{TemplateEngine, TemplateError};

/// Format chat messages using the model's chat template or architecture fallback.
///
/// # Arguments
/// * `messages` - Chat messages to format
/// * `template` - Chat template from GGUF (if available)
/// * `architecture` - Model architecture name for fallback template selection
/// * `model_name` - Model name for detecting instruct vs base models
/// * `bos_token` - BOS token string
/// * `eos_token` - EOS token string
/// * `add_generation_prompt` - Whether to add assistant prompt at the end
///
/// # Returns
/// Tuple of (formatted prompt, stop strings)
pub fn format_messages(
    messages: &[SimpleMessage],
    template: Option<&str>,
    architecture: &str,
    model_name: Option<&str>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    add_generation_prompt: bool,
) -> Result<(String, Vec<String>), TemplateError> {
    let engine = TemplateEngine::new(template, architecture, model_name, bos_token, eos_token)?;
    let result = engine.render(messages, add_generation_prompt)?;
    Ok((result.prompt, result.stop_strings))
}

/// Format messages with sensible defaults for missing info.
pub fn format_messages_with_defaults(
    messages: &[SimpleMessage],
    template: Option<&str>,
    architecture: &str,
    model_name: Option<&str>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    add_generation_prompt: bool,
) -> (String, Vec<String>) {
    match format_messages(
        messages,
        template,
        architecture,
        model_name,
        bos_token,
        eos_token,
        add_generation_prompt,
    ) {
        Ok((prompt, stops)) => (prompt, stops),
        Err(e) => {
            tracing::warn!(error = %e, "Template rendering failed, using simple fallback");
            // Fallback to simple role: content format
            let mut prompt = String::new();
            for msg in messages {
                prompt.push_str(&msg.role);
                prompt.push_str(": ");
                prompt.push_str(msg.content.trim());
                prompt.push('\n');
            }
            if add_generation_prompt {
                prompt.push_str("assistant: ");
            }
            (prompt, Vec::new())
        }
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
    fn test_format_with_architecture() {
        let messages = make_messages();
        // Use instruct model name to trigger architecture fallback
        let (prompt, stops) = format_messages(
            &messages,
            None,
            "llama",
            Some("llama-3-instruct"),
            Some("<|begin_of_text|>".to_string()),
            None,
            true,
        )
        .unwrap();

        assert!(prompt.contains("<|start_header_id|>"));
        assert!(prompt.contains("<|eot_id|>"));
        assert!(stops.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_format_with_defaults_unknown_arch() {
        let messages = make_messages();
        // Base model (no instruct indicator) uses default format (raw prompt only)
        let (prompt, _stops) =
            format_messages_with_defaults(&messages, None, "unknown", Some("base-model"), None, None, true);

        // With base model name, default template outputs only the last user message
        assert!(prompt.contains("How are you?"));
        assert!(!prompt.contains("<|")); // No special tokens
    }

    #[test]
    fn test_format_with_custom_template() {
        let template = r#"
{%- for message in messages %}
[{{ message.role }}]: {{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}
[assistant]:
{%- endif %}
"#;
        let messages = make_messages();
        // Custom template takes priority regardless of model name
        let (prompt, _) = format_messages(&messages, Some(template), "unknown", None, None, None, true).unwrap();

        assert!(prompt.contains("[user]: Hello!"));
        assert!(prompt.contains("[assistant]: Hi there!"));
    }
}
