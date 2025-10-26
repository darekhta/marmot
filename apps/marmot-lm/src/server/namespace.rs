//! LM Studio API namespaces (WebSocket paths).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiNamespace {
    Llm,
    System,
    Embedding,
    Diagnostics,
    Files,
    Repository,
    Plugins,
    Runtime,
    Unknown,
}

impl ApiNamespace {
    pub fn parse(namespace: &str) -> Self {
        match namespace {
            "llm" => Self::Llm,
            "system" => Self::System,
            "embedding" => Self::Embedding,
            "diagnostics" => Self::Diagnostics,
            "files" => Self::Files,
            "repository" => Self::Repository,
            "plugins" => Self::Plugins,
            "runtime" => Self::Runtime,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::System => "system",
            Self::Embedding => "embedding",
            Self::Diagnostics => "diagnostics",
            Self::Files => "files",
            Self::Repository => "repository",
            Self::Plugins => "plugins",
            Self::Runtime => "runtime",
            Self::Unknown => "unknown",
        }
    }
}
