//! Stop string resolution for chat templates.

use super::fallback::{self, FallbackRegistry};

/// Stop strings by template family.
const STOP_STRINGS: &[(&str, &[&str])] = &[
    (fallback::FAMILY_LLAMA3, &["<|eot_id|>", "<|end_of_text|>"]),
    (fallback::FAMILY_CHATML, &["<|im_end|>"]),
    (fallback::FAMILY_GEMMA, &["<end_of_turn>"]),
    (fallback::FAMILY_PHI3, &["<|end|>", "<|user|>"]),
    (fallback::FAMILY_MISTRAL, &["</s>", "[/INST]"]),
    // Default format for base models - stop on newlines or common end patterns
    (fallback::FAMILY_DEFAULT, &["\n\n", "\n---"]),
];

/// Resolver for stop strings.
pub struct StopStringResolver;

impl StopStringResolver {
    /// Get stop strings for a template family.
    pub fn for_family(family: &str) -> Vec<String> {
        for (fam, stops) in STOP_STRINGS {
            if *fam == family {
                return stops.iter().map(|s| (*s).to_string()).collect();
            }
        }
        Vec::new()
    }

    /// Get stop strings for an architecture.
    pub fn for_architecture(architecture: &str) -> Vec<String> {
        let family = FallbackRegistry::get_family(architecture);
        Self::for_family(family)
    }

    /// Extract stop strings from template content by looking for common patterns.
    pub fn from_template(template: &str) -> Vec<String> {
        let mut stops = Vec::new();

        // Common stop token patterns
        // These include both EOS-style tokens and role markers that indicate
        // the model is trying to prompt for another turn (e.g., <|user|>)
        let patterns = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<end_of_turn>",
            "<|end|>",
            "</s>",
            "[/INST]",
            "<|endoftext|>",
            // Role markers - stop if model outputs these (prevents runaway generation)
            "<|user|>",
            "<|system|>",
        ];

        for pattern in patterns {
            if template.contains(pattern) {
                stops.push(pattern.to_string());
            }
        }

        stops
    }

    /// Resolve stop strings from template or architecture.
    ///
    /// When a template is provided, we extract stop strings from it directly.
    /// Architecture defaults are only used when no template is available.
    pub fn resolve(template: Option<&str>, architecture: &str) -> Vec<String> {
        if let Some(tmpl) = template {
            // Template is the source of truth - extract stops from it
            let mut stops = Self::from_template(tmpl);
            if stops.is_empty() && tmpl.trim() == fallback::DEFAULT_TEMPLATE.trim() {
                stops = Self::for_family(fallback::FAMILY_DEFAULT);
            }
            return stops;
        } else {
            // No template - use architecture defaults
            Self::for_architecture(architecture)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_stops() {
        let stops = StopStringResolver::for_family(fallback::FAMILY_LLAMA3);
        assert!(stops.contains(&"<|eot_id|>".to_string()));

        let stops = StopStringResolver::for_family(fallback::FAMILY_CHATML);
        assert!(stops.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_from_template() {
        let template = "{{- '<|im_start|>' }}...{{- '<|im_end|>' }}";
        let stops = StopStringResolver::from_template(template);
        assert!(stops.contains(&"<|im_end|>".to_string()));
    }

    #[test]
    fn test_resolve() {
        // With template
        let stops = StopStringResolver::resolve(Some("<|im_end|>"), "unknown");
        assert!(stops.contains(&"<|im_end|>".to_string()));

        // Without template, use arch
        let stops = StopStringResolver::resolve(None, "llama");
        assert!(stops.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_resolve_default_template_adds_stops() {
        let stops = StopStringResolver::resolve(Some(fallback::DEFAULT_TEMPLATE), "gemma");
        assert!(stops.contains(&"\n\n".to_string()));
    }
}
