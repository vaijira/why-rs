#![warn(
    rust_2018_idioms,
    missing_docs,
    missing_debug_implementations,
    unused_extern_crates,
    warnings
)]

//! Causal data structures.

// Documented by its own `//!` header.
pub mod abstraction;
// Documented by its own `//!` header.
pub mod ascm;
// Documented by its own `//!` header.
pub mod crl;
// Documented by its own `//!` header.
pub mod discovery;
// Documented by its own `//!` header.
pub mod explain;
// Documented by its own `//!` header.
pub mod fairness;
// Documented by its own `//!` header.
pub mod generalize;
/// Causal graph support
pub mod graph;
// Documented by its own `//!` header.
pub mod ncm;
// Documented by its own `//!` header.
pub mod representation;
// Documented by its own `//!` header, which is where the intra-doc links
// resolve from; an outer `///` here would shadow it and break them.
pub mod scm;
// Documented by its own `//!` header.
pub mod transport;
/// Different useful types
pub mod types;
