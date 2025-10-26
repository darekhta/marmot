//! Chat template rendering using Jinja2 (minijinja).
//!
//! This module provides real Jinja2 template execution for chat formatting,
//! with architecture-based fallback templates when GGUF doesn't provide one.

mod context;
mod engine;
mod fallback;
mod filters;
mod source;
mod stop_strings;

pub use engine::{TemplateEngine, TemplateError};

/// Result of rendering a chat template.
#[derive(Debug, Clone)]
pub struct RenderResult {
    /// The formatted prompt string.
    pub prompt: String,
    /// Stop strings for this template/model.
    pub stop_strings: Vec<String>,
}
