//! Template source abstraction.
//!
//! This trait allows for future extensibility to support different
//! model formats (safetensors, etc.) as template sources.

use super::fallback::{FallbackRegistry, DEFAULT_TEMPLATE};

/// Abstraction for template sources.
#[allow(dead_code)]
pub trait TemplateSource: Send + Sync {
    /// Get the template string.
    fn template(&self) -> &str;

    /// Get source type for logging/debugging.
    fn source_type(&self) -> &'static str;
}

/// Template loaded from GGUF metadata.
pub struct GgufTemplateSource {
    template: String,
}

impl GgufTemplateSource {
    pub fn new(template: String) -> Self {
        Self { template }
    }
}

impl TemplateSource for GgufTemplateSource {
    fn template(&self) -> &str {
        &self.template
    }

    fn source_type(&self) -> &'static str {
        "gguf"
    }
}

/// Fallback template based on architecture (for instruct-tuned models).
pub struct FallbackTemplateSource {
    template: &'static str,
}

impl FallbackTemplateSource {
    pub fn new(architecture: &str) -> Self {
        Self {
            template: FallbackRegistry::get_template(architecture),
        }
    }
}

impl TemplateSource for FallbackTemplateSource {
    fn template(&self) -> &str {
        self.template
    }

    fn source_type(&self) -> &'static str {
        "fallback"
    }
}

/// Default template for base models (simple role: content format).
pub struct DefaultTemplateSource;

impl DefaultTemplateSource {
    pub fn new() -> Self {
        Self
    }
}

impl TemplateSource for DefaultTemplateSource {
    fn template(&self) -> &str {
        DEFAULT_TEMPLATE
    }

    fn source_type(&self) -> &'static str {
        "default"
    }
}

/// Checks if model name suggests it's an instruct-tuned model.
fn is_likely_instruct_model(model_name: Option<&str>) -> bool {
    let Some(name) = model_name else {
        return false;
    };
    let name_lower = name.to_lowercase();

    // Check for common instruct model indicators
    name_lower.contains("chat")
        || name_lower.contains("instruct")
        || name_lower.contains("-it")
        || name_lower.ends_with("it")
        || name_lower.contains("assistant")
}

/// Select the appropriate template source.
///
/// Priority:
/// 1. GGUF template if available and non-empty
/// 2. Architecture fallback if model appears to be instruct-tuned
/// 3. Default format for base models
pub fn select_source(
    gguf_template: Option<&str>,
    architecture: &str,
    model_name: Option<&str>,
) -> Box<dyn TemplateSource> {
    // 1. Use GGUF template if available
    if let Some(tmpl) = gguf_template {
        if !tmpl.trim().is_empty() {
            tracing::debug!(source = "gguf", "Using GGUF chat template");
            return Box::new(GgufTemplateSource::new(tmpl.to_string()));
        }
    }

    // 2. Use architecture fallback for instruct models
    if is_likely_instruct_model(model_name) {
        let family = FallbackRegistry::get_family(architecture);
        tracing::debug!(
            source = "fallback",
            architecture = %architecture,
            family = %family,
            model_name = ?model_name,
            "Model appears to be instruct-tuned, using architecture fallback"
        );
        return Box::new(FallbackTemplateSource::new(architecture));
    }

    // 3. Use default format for base models
    let family = FallbackRegistry::get_family(architecture);
    if family != super::fallback::FAMILY_DEFAULT {
        tracing::warn!(
            architecture = %architecture,
            family = %family,
            model_name = ?model_name,
            "Model architecture '{}' is a known chat family ('{}') but no GGUF template found \
             and model name does not indicate instruct tuning. Using base-model default template. \
             This likely produces poor output. Consider renaming the model to include 'instruct' \
             or 'chat', or set the GGUF chat_template metadata.",
            architecture, family
        );
    } else {
        tracing::debug!(
            source = "default",
            model_name = ?model_name,
            "Model appears to be base model, using default format"
        );
    }
    Box::new(DefaultTemplateSource::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_source() {
        let source = GgufTemplateSource::new("{{ messages }}".to_string());
        assert_eq!(source.template(), "{{ messages }}");
        assert_eq!(source.source_type(), "gguf");
    }

    #[test]
    fn test_fallback_source() {
        let source = FallbackTemplateSource::new("llama");
        assert!(source.template().contains("<|start_header_id|>"));
        assert_eq!(source.source_type(), "fallback");
    }

    #[test]
    fn test_default_source() {
        let source = DefaultTemplateSource::new();
        // Default template outputs only the last user message (minimal format for base models)
        assert!(source.template().contains("last_user_msg"));
        assert_eq!(source.source_type(), "default");
    }

    #[test]
    fn test_select_source_prefers_gguf() {
        // GGUF template takes priority regardless of model name
        let source = select_source(Some("custom template"), "llama", Some("llama-base"));
        assert_eq!(source.source_type(), "gguf");
        assert_eq!(source.template(), "custom template");
    }

    #[test]
    fn test_select_source_fallback_for_instruct() {
        // No GGUF template + instruct model name = architecture fallback
        let source = select_source(None, "gemma", Some("gemma-2b-it"));
        assert_eq!(source.source_type(), "fallback");
        assert!(source.template().contains("<start_of_turn>"));

        let source = select_source(None, "llama", Some("TinyLlama-1.1B-Chat-v1.0"));
        assert_eq!(source.source_type(), "fallback");
        assert!(source.template().contains("<|start_header_id|>"));

        let source = select_source(None, "qwen2", Some("Qwen2-0.5B-Instruct"));
        assert_eq!(source.source_type(), "fallback");
    }

    #[test]
    fn test_select_source_default_for_base() {
        // No GGUF template + base model name = default format
        let source = select_source(None, "gemma", Some("gemma-2b"));
        assert_eq!(source.source_type(), "default");

        let source = select_source(None, "llama", Some("llama-7b"));
        assert_eq!(source.source_type(), "default");

        // No model name = treat as base model (conservative)
        let source = select_source(None, "gemma", None);
        assert_eq!(source.source_type(), "default");
    }

    #[test]
    fn test_known_chat_arch_with_base_name_uses_default() {
        // Known chat architecture (qwen2) but base model name => default source
        // This scenario triggers the warning in production
        let source = select_source(None, "qwen2", Some("qwen2-7b"));
        assert_eq!(source.source_type(), "default");

        let source = select_source(None, "gemma", Some("gemma-2b"));
        assert_eq!(source.source_type(), "default");

        let source = select_source(None, "llama", Some("llama-7b"));
        assert_eq!(source.source_type(), "default");
    }

    #[test]
    fn test_is_likely_instruct_model() {
        // Positive cases
        assert!(is_likely_instruct_model(Some("TinyLlama-1.1B-Chat-v1.0")));
        assert!(is_likely_instruct_model(Some("gemma-2b-it")));
        assert!(is_likely_instruct_model(Some("Qwen2-0.5B-Instruct")));
        assert!(is_likely_instruct_model(Some("llama-3-8b-instruct")));
        assert!(is_likely_instruct_model(Some("CodeAssistant-7B")));

        // Negative cases
        assert!(!is_likely_instruct_model(Some("gemma-2b")));
        assert!(!is_likely_instruct_model(Some("llama-7b")));
        assert!(!is_likely_instruct_model(Some("phi-2")));
        assert!(!is_likely_instruct_model(None));
    }
}
