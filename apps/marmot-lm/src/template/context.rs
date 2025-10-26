//! Render context for Jinja2 templates.

use minijinja::Value;
use serde::Serialize;

use crate::common::SimpleMessage;

/// A message in the format expected by Jinja2 templates.
#[derive(Debug, Clone, Serialize)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

impl From<&SimpleMessage> for TemplateMessage {
    fn from(msg: &SimpleMessage) -> Self {
        Self {
            role: msg.role.clone(),
            content: msg.content.clone(),
        }
    }
}

/// Context passed to Jinja2 template rendering.
#[derive(Debug, Clone)]
pub struct RenderContext {
    /// Chat messages to render.
    pub messages: Vec<TemplateMessage>,
    /// Whether to add assistant generation prompt at the end.
    pub add_generation_prompt: bool,
    /// BOS token string (e.g., "<s>", "<|begin_of_text|>").
    pub bos_token: String,
    /// EOS token string (e.g., "</s>", "<|end_of_text|>").
    pub eos_token: String,
}

impl RenderContext {
    /// Create a new render context from messages.
    pub fn new(
        messages: &[SimpleMessage],
        add_generation_prompt: bool,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self {
            messages: messages.iter().map(TemplateMessage::from).collect(),
            add_generation_prompt,
            bos_token: bos_token.unwrap_or_default(),
            eos_token: eos_token.unwrap_or_default(),
        }
    }

    /// Convert to minijinja Value for template rendering.
    pub fn to_value(&self) -> Value {
        Value::from_serialize(ContextValue {
            messages: &self.messages,
            add_generation_prompt: self.add_generation_prompt,
            bos_token: &self.bos_token,
            eos_token: &self.eos_token,
        })
    }
}

#[derive(Serialize)]
struct ContextValue<'a> {
    messages: &'a [TemplateMessage],
    add_generation_prompt: bool,
    bos_token: &'a str,
    eos_token: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let messages = vec![
            SimpleMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            SimpleMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
        ];

        let ctx = RenderContext::new(&messages, true, Some("<s>".to_string()), Some("</s>".to_string()));

        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role, "user");
        assert!(ctx.add_generation_prompt);
        assert_eq!(ctx.bos_token, "<s>");
        assert_eq!(ctx.eos_token, "</s>");
    }
}
