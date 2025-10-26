//! FFI bindings to libmarmot
//!
//! This module provides both raw C bindings and safe Rust wrappers for the marmot library.

pub mod bindings;
pub mod safe_wrappers;

pub use safe_wrappers::*;
