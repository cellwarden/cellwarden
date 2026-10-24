//! Shared data types. Inputs are borrowed views so R (`dgCMatrix`) and Python (scipy CSR)
//! data are read in place, without copying or transposing (docs/RUST_ENGINE.md section 3).

use rayon::prelude::*;

use crate::error::{Error, Result};

/// Integer types used for sparse indices: R uses i32, scipy uses i32 or i64.
pub trait Idx: Copy + Send + Sync + 'static {
    fn to_usize(self) -> usize;
}
macro_rules! impl_idx {
    ($($t:ty),*) => { $(impl Idx for $t { #[inline(always)] fn to_usize(self) -> usize { self as usize } })* };
}
impl_idx!(i32, i64, u32, u64, usize);

/// Value types: R `dgCMatrix` stores f64, AnnData usually f32.
pub trait Real: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;
}
impl Real for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self as f64 }
    #[inline(always)]
    fn to_f32(self) -> f32 { self }
}
impl Real for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 { self }
    #[inline(always)]
    fn to_f32(self) -> f32 { self as f32 }
}

/// Borrowed CSR matrix, cells x genes.
///
/// A Seurat `dgCMatrix` of genes x cells (CSC) has exactly these arrays: its `@p` is
/// `indptr`, `@i` is `indices`, `@x` is `data`. A scipy `csr_matrix` of cells x genes maps
/// directly too.
#[derive(Clone, Copy, Debug)]
pub struct CsrView<'a, I: Idx, T: Real> {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: &'a [I],
    pub indices: &'a [I],
    pub data: &'a [T],
}

impl<'a, I: Idx, T: Real> CsrView<'a, I, T> {
    /// Validates shapes and index bounds (O(nnz)); does not require sorted indices.
    pub fn new(n_rows: usize, n_cols: usize, indptr: &'a [I], indices: &'a [I], data: &'a [T]) -> Result<Self> {
        if indptr.len() != n_rows + 1 {
            return Err(Error::InvalidInput(format!("indptr has {} entries, expected n_rows + 1 = {}", indptr.len(), n_rows + 1)));
        }
        if indices.len() != data.len() {
            return Err(Error::InvalidInput(format!("indices ({}) and data ({}) differ in length", indices.len(), data.len())));
        }
        if indptr[0].to_usize() != 0 || indptr[n_rows].to_usize() != data.len() {
            return Err(Error::InvalidInput("indptr must start at 0 and end at nnz".into()));
        }
        for w in indptr.windows(2) {
            if w[1].to_usize() < w[0].to_usize() {
                return Err(Error::InvalidInput("indptr is not non-decreasing".into()));
            }
        }
        let out_of_range = |c: &&[I]| c.iter().any(|j| j.to_usize() >= n_cols);
        let bad_chunk = indices.par_chunks(1 << 16).find_first(out_of_range);
        if let Some(bad) = bad_chunk.and_then(|c| c.iter().find(|&&j| j.to_usize() >= n_cols)) {
            return Err(Error::InvalidInput(format!("column index {} out of range for {} columns", bad.to_usize(), n_cols)));
        }
        Ok(Self { n_rows, n_cols, indptr, indices, data })
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// (column indices, values) of one row.
    #[inline]
    pub fn row(&self, i: usize) -> (&'a [I], &'a [T]) {
        let (s, e) = (self.indptr[i].to_usize(), self.indptr[i + 1].to_usize());
        (&self.indices[s..e], &self.data[s..e])
    }
}

/// Dense row-major f32 matrix: embeddings (cells x components) and loadings.
#[derive(Clone, Debug, PartialEq)]
pub struct RowMajor {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<f32>,
}

impl RowMajor {
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        Self { n_rows, n_cols, data: vec![0.0; n_rows * n_cols] }
    }
    pub fn from_vec(n_rows: usize, n_cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != n_rows * n_cols {
            return Err(Error::InvalidInput(format!("{} values for a {}x{} matrix", data.len(), n_rows, n_cols)));
        }
        Ok(Self { n_rows, n_cols, data })
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_cols..(i + 1) * self.n_cols]
    }
}

/// k nearest neighbours of every cell, row-major n x k.
///
/// Convention (matches Seurat's `k.param`): `k` counts the cell itself, and `indices[i*k]`
/// is `i` with distance 0 whenever `include_self` was requested. Distances are Euclidean.
#[derive(Clone, Debug)]
pub struct Knn {
    pub n: usize,
    pub k: usize,
    pub indices: Vec<u32>,
    pub distances: Vec<f32>,
}

/// Weighted undirected graph as a full symmetric CSR adjacency (both directions of every
/// edge stored): the layout `single_clustering::network::CSRNetwork::from_csr_parts` takes,
/// and convertible to a Seurat `Graph` / scipy CSR without reshaping.
#[derive(Clone, Debug)]
pub struct SparseGraph {
    pub n: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

impl SparseGraph {
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}
