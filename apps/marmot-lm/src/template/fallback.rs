//! Architecture-based fallback templates.
//!
//! Provides embedded Jinja2 templates for common model architectures,
//! used when the GGUF file doesn't include a chat template.

use std::env;

/// Embedded template files.
const LLAMA3_TEMPLATE: &str = include_str!("templates/llama3.jinja");
const CHATML_TEMPLATE: &str = include_str!("templates/chatml.jinja");
const GEMMA_TEMPLATE: &str = include_str!("templates/gemma.jinja");
const PHI3_TEMPLATE: &str = include_str!("templates/phi3.jinja");
const MISTRAL_TEMPLATE: &str = include_str!("templates/mistral.jinja");
pub const DEFAULT_TEMPLATE: &str = include_str!("templates/default.jinja");

/// Template family names.
pub const FAMILY_LLAMA3: &str = "llama3";
pub const FAMILY_CHATML: &str = "chatml";
pub const FAMILY_GEMMA: &str = "gemma";
pub const FAMILY_PHI3: &str = "phi3";
pub const FAMILY_MISTRAL: &str = "mistral";
pub const FAMILY_DEFAULT: &str = "default";

/// Architecture-to-template-family mapping.
/// NOTE: More specific entries (e.g., llama2, llama3) must come BEFORE
/// less specific ones (e.g., llama) since lookup uses starts_with.
const ARCHITECTURE_MAP: &[(&str, &str)] = &[
    // Llama family - specific versions first!
    ("llama3", FAMILY_LLAMA3),
    ("llama2", FAMILY_MISTRAL), // Llama 2 uses [INST] format
    ("llama", FAMILY_LLAMA3),   // Generic llama defaults to llama3
    // Mistral/Mixtral (use llama3 format for modern versions)
    ("mistral", FAMILY_LLAMA3),
    ("mixtral", FAMILY_LLAMA3),
    // Qwen family (ChatML) - specific versions first
    ("qwen2moe", FAMILY_CHATML),
    ("qwen2", FAMILY_CHATML),
    ("qwen3", FAMILY_CHATML),
    ("qwen", FAMILY_CHATML),
    // Yi (ChatML)
    ("yi", FAMILY_CHATML),
    // DeepSeek (ChatML) - specific versions first
    ("deepseek2", FAMILY_CHATML),
    ("deepseek", FAMILY_CHATML),
    // Gemma family - specific versions first
    ("gemma2", FAMILY_GEMMA),
    ("gemma", FAMILY_GEMMA),
    // Phi family - specific versions first
    ("phi3", FAMILY_PHI3),
    ("phi2", FAMILY_PHI3),
    ("phi", FAMILY_PHI3),
    // StarCoder (ChatML-like)
    ("starcoder2", FAMILY_CHATML),
    ("starcoder", FAMILY_CHATML),
    // InternLM (ChatML)
    ("internlm2", FAMILY_CHATML),
    ("internlm", FAMILY_CHATML),
    // Command-R (ChatML variant)
    ("command-r", FAMILY_CHATML),
];

/// Registry for fallback templates.
pub struct FallbackRegistry;

impl FallbackRegistry {
    /// Get the template family name for an architecture.
    pub fn get_family(architecture: &str) -> &'static str {
        // Check environment override first
        let env_key = format!(
            "MARMOT_TEMPLATE_OVERRIDE_{}",
            architecture.to_lowercase().replace('-', "_")
        );
        if let Ok(override_family) = env::var(&env_key) {
            return Self::family_to_static(&override_family);
        }

        // Normalize architecture name
        let arch_lower = architecture.to_lowercase();
        let arch_normalized = arch_lower
            .trim_start_matches("marmot-")
            .trim_end_matches("-hf")
            .replace(['-', '_'], "");

        // Look up in mapping
        for (arch, family) in ARCHITECTURE_MAP {
            if arch_normalized.starts_with(arch) || arch_normalized == *arch {
                return family;
            }
        }

        FAMILY_DEFAULT
    }

    /// Get the template string for an architecture.
    pub fn get_template(architecture: &str) -> &'static str {
        // Check for custom template path override
        if let Ok(custom_path) = env::var("MARMOT_TEMPLATE_PATH") {
            // For custom paths, we can't return a static str, so fall back to family
            tracing::warn!(
                path = %custom_path,
                "MARMOT_TEMPLATE_PATH set but custom file loading not yet implemented, using family fallback"
            );
        }

        let family = Self::get_family(architecture);
        Self::template_for_family(family)
    }

    /// Get template content for a family name.
    pub fn template_for_family(family: &str) -> &'static str {
        match family {
            FAMILY_LLAMA3 => LLAMA3_TEMPLATE,
            FAMILY_CHATML => CHATML_TEMPLATE,
            FAMILY_GEMMA => GEMMA_TEMPLATE,
            FAMILY_PHI3 => PHI3_TEMPLATE,
            FAMILY_MISTRAL => MISTRAL_TEMPLATE,
            _ => DEFAULT_TEMPLATE,
        }
    }

    /// Convert a runtime family string to a static reference.
    fn family_to_static(family: &str) -> &'static str {
        match family.to_lowercase().as_str() {
            "llama3" | "llama" => FAMILY_LLAMA3,
            "chatml" | "qwen" => FAMILY_CHATML,
            "gemma" => FAMILY_GEMMA,
            "phi3" | "phi" => FAMILY_PHI3,
            "mistral" | "llama2" => FAMILY_MISTRAL,
            _ => FAMILY_DEFAULT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_mapping() {
        assert_eq!(FallbackRegistry::get_family("llama"), FAMILY_LLAMA3);
        assert_eq!(FallbackRegistry::get_family("Llama"), FAMILY_LLAMA3);
        assert_eq!(FallbackRegistry::get_family("qwen2"), FAMILY_CHATML);
        assert_eq!(FallbackRegistry::get_family("gemma"), FAMILY_GEMMA);
        assert_eq!(FallbackRegistry::get_family("phi3"), FAMILY_PHI3);
        assert_eq!(FallbackRegistry::get_family("unknown"), FAMILY_DEFAULT);
    }

    #[test]
    fn test_template_retrieval() {
        let template = FallbackRegistry::get_template("llama");
        assert!(template.contains("<|start_header_id|>"));

        let template = FallbackRegistry::get_template("qwen2");
        assert!(template.contains("<|im_start|>"));

        let template = FallbackRegistry::get_template("gemma");
        assert!(template.contains("<start_of_turn>"));
    }
}
