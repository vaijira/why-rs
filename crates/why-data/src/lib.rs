#![warn(
    rust_2018_idioms,
    missing_docs,
    missing_debug_implementations,
    unused_extern_crates,
    warnings
)]

//! Causal data structures.

// Documented by its own `//!` header.
pub mod crl;
// Documented by its own `//!` header.
pub mod explain;
// Documented by its own `//!` header.
pub mod fairness;
/// Causal graph support
pub mod graph;
// Documented by its own `//!` header, which is where the intra-doc links
// resolve from; an outer `///` here would shadow it and break them.
pub mod scm;
/// Different useful types
pub mod types;
