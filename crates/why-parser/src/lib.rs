#![warn(
    rust_2018_idioms,
    missing_docs,
    missing_debug_implementations,
    unused_extern_crates,
    warnings
)]

//! Parsers for causal model formats.

/// Parsing module.
pub mod dagitty;
// Documented by its own `//!` header, which is where the intra-doc links
// resolve from; an outer `///` here would shadow it and break them.
pub mod fusion;
