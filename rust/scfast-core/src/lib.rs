//! scfast-core: the numerical core of the scfast engine (docs/RUST_ENGINE.md).
//!
//! Kernels, each callable on its own so Seurat (via scfast-r) and Python (via scfast-py)
//! can swap in one step at a time:
//!
//! | Module        | Replaces                      | Function                     |
//! |---------------|-------------------------------|------------------------------|
//! | [`pca`]       | `ScaleData` + `RunPCA`        | [`pca::implicit_scaled_pca`] |
//! | [`knn`]       | kNN part of `FindNeighbors`   | [`knn::knn`]                 |
//! | [`snn`]       | SNN part of `FindNeighbors`   | [`snn::snn_from_knn`]        |
//! | [`cluster`]   | `FindClusters(algorithm = 4)` | [`cluster::leiden`]          |
//! | [`integrate`] | `RunHarmony` (harmony 2.x)    | [`integrate::harmony()`]     |
//!
//! Everything here is the CPU reference implementation; the CUDA backend (Phase 6) is
//! tested against it. GPU Leiden ([`cugraph`], feature `cugraph`, Phase 7) goes through
//! cuGraph's C API loaded at runtime and is tested against [`cluster::SingleClusteringLeiden`].

pub mod cluster;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cugraph")]
pub mod cugraph;
pub mod device;
pub mod error;
pub mod integrate;
pub mod knn;
pub mod pca;
pub mod preprocess;
pub mod snn;
pub mod types;
pub mod umap;

pub use device::{device_info, Device, DeviceInfo};
pub use error::{Error, Result};
pub use types::{CsrView, Idx, Knn, Real, RowMajor, SparseGraph};
