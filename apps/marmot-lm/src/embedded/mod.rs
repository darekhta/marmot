//! Embedded inference mode (no server)
//!
//! Thin wrapper around libmarmot - no duplicate logic.

pub mod runtime;

pub use runtime::EmbeddedRuntime;
