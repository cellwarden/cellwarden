//! Batch integration of cell embeddings.
//!
//! [`harmony`](harmony::harmony) replaces `harmony::RunHarmony()` (R) and
//! `harmonypy.run_harmony()` on a PCA embedding; see [`mod@harmony`] for the algorithm and how
//! it follows the R package. CPU only for now: `Device::Cuda` returns [`crate::Error::Gpu`].

mod fastmath;
pub mod harmony;
mod kmeans;

pub use harmony::{harmony, harmony_dense, HarmonyParams, HarmonyResult, Layout};
