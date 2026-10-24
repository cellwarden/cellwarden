//! kNN on a CUDA device (feature `cuda`): the fused `knn` kernel (scores and per-query
//! top-k in one pass, see `cuda/mod.rs`), with the same exact f64 re-ranking and output
//! contract as the CPU methods.
//!
//! Data are centred on the host (f64 mean, rounded to f32) and uploaded once with their
//! squared norms. Queries go in blocks of `qb` (bounded by the candidate buffer's memory);
//! per block the kernel keeps the `cap = k - 1 + slack` smallest f32 estimates
//! `|r|^2 - 2 q.r` per query, and the host re-ranks those candidates by exact f64
//! distances (`exact_d2`, as `exact.rs`) and writes the row.
//!
//! The error-bound certificate of the CPU exact search is not applied: the result is exact
//! whenever the true k - 1 neighbours are among the `cap` best f32 estimates, which holds
//! for all but pathological ties; the parity target for the GPU path is a neighbour
//! overlap of at least 90% with the CPU result. Values so large that the f32 estimates
//! could overflow are handed to the CPU exact search.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use rayon::prelude::*;

use super::{cmp_neighbour, exact_d2, write_row};
use crate::cuda::{drv_err, guard, Gpu, KNN_BLOCK, KNN_CAP_MAX, KNN_QPT};
use crate::error::{Error, Result};
use crate::types::{Knn, RowMajor};

/// Queries per kernel launch, at most (bounds the candidate buffers: 8 * cap bytes each).
const MAX_QB: usize = 1 << 20;
/// Same overflow guard as `exact.rs`.
const F32_SAFE_SQ_NORM: f64 = 1e36;

pub(super) fn cuda_knn(x: &RowMajor, k: usize, ordinal: usize) -> Result<Knn> {
    let (n, d) = (x.n_rows, x.n_cols);
    let kk = k - 1;
    if n > i32::MAX as usize || d > i32::MAX as usize {
        return Err(Error::gpu(format!("{n} cells x {d} dims exceeds the CUDA kernel index range")));
    }
    if kk > KNN_CAP_MAX {
        return Err(Error::gpu(format!("k = {k} is above the CUDA kNN limit of {} (k counts the cell itself); use the CPU", KNN_CAP_MAX + 1)));
    }
    let slack = (kk / 2).max(8);
    let cap = (kk + slack).min(n - 1).min(KNN_CAP_MAX).max(kk);

    // Centre (f64 mean, rounded to f32), as the CPU exact search does.
    let mut mean = vec![0f64; d];
    for i in 0..n {
        for (m, &v) in mean.iter_mut().zip(x.row(i)) {
            *m += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    // Rows go to the device in locality order (`perm[s]` is the row at position s), which
    // the kernel's scan order relies on; candidates are mapped back on the host.
    let perm = morton_order(x, &mean);
    let mut xc = vec![0f32; n * d];
    let mut norms = vec![0f32; n];
    xc.par_chunks_mut(d).zip(norms.par_iter_mut()).zip(perm.par_iter()).for_each(|((row, nrm), &i)| {
        let mut s = 0f64;
        for (t, (y, &v)) in row.iter_mut().zip(x.row(i as usize)).enumerate() {
            *y = (v as f64 - mean[t]) as f32;
            s += *y as f64 * *y as f64;
        }
        *nrm = s as f32;
    });
    let max_norm = norms.iter().fold(0f64, |m, &v| m.max(v as f64));
    if max_norm * 4.0 > F32_SAFE_SQ_NORM {
        // f32 cross terms could overflow: the CPU exact search handles this case.
        return Ok(super::exact::exact_knn(x, k));
    }

    let gpu = Gpu::get(ordinal)?;
    let base = 4 * n * (d + 1);
    gpu.ensure_fits(base + 4 * cap, &format!("kNN on {n} cells x {d} dims"))?;
    let free = gpu.free_bytes()?;
    let qb = (free.saturating_sub(base) / 2 / (4 * cap)).min(MAX_QB).min(n);
    if qb == 0 {
        return Err(Error::gpu(format!("out of device memory: kNN on {n} cells, {:.2} GiB free", free as f64 / (1u64 << 30) as f64)));
    }

    let cand = candidates(&gpu, &xc, &norms, n, d, cap, qb)?;


    // Exact re-ranking of the candidates on the host, in the original row order.
    let mut inv = vec![0u32; n];
    for (s, &i) in perm.iter().enumerate() {
        inv[i as usize] = s as u32;
    }
    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];
    let fallback: usize = indices
        .par_chunks_mut(k)
        .zip(distances.par_chunks_mut(k))
        .enumerate()
        .map_init(
            || Vec::with_capacity(cap),
            |scratch, (i, (irow, drow))| {
                let xi = x.row(i);
                let s = inv[i] as usize;
                scratch.clear();
                for &js in &cand[s * cap..(s + 1) * cap] {
                    if js != u32::MAX {
                        let j = perm[js as usize];
                        if j as usize != i {
                            scratch.push((exact_d2(xi, x.row(j as usize)), j));
                        }
                    }
                }
                let mut fb = 0;
                if scratch.len() < kk {
                    // Cannot happen when cap <= n - 1; kept as a safety net.
                    super::brute_one(x, i, kk, scratch);
                    fb = 1;
                }
                scratch.sort_unstable_by(cmp_neighbour);
                write_row(i, scratch, irow, drow);
                fb
            },
        )
        .sum();
    debug_assert_eq!(fallback, 0);
    Ok(Knn { n, k, indices, distances })
}

/// Per run of `len` consecutive rows: the box (lo[d], hi[d]) around them.
fn boxes(xc: &[f32], d: usize, len: usize) -> Vec<f32> {
    xc.par_chunks(len * d)
        .flat_map_iter(|run| {
            let mut b = vec![f32::INFINITY; d];
            b.extend(std::iter::repeat_n(f32::NEG_INFINITY, d));
            for row in run.chunks(d) {
                for (t, &v) in row.iter().enumerate() {
                    b[t] = b[t].min(v);
                    b[d + t] = b[d + t].max(v);
                }
            }
            b
        })
        .collect()
}

/// Rows sorted along a Z-order (Morton) curve over the three widest dimensions, 10 bits
/// each: rows close in that order are mostly close in space.
fn morton_order(x: &RowMajor, mean: &[f64]) -> Vec<u32> {
    let (n, d) = (x.n_rows, x.n_cols);
    let mut var = vec![0f64; d];
    for i in 0..n {
        for ((v, &y), &m) in var.iter_mut().zip(x.row(i)).zip(mean) {
            *v += (y as f64 - m) * (y as f64 - m);
        }
    }
    let mut dims: Vec<usize> = (0..d).collect();
    dims.sort_by(|&a, &b| var[b].total_cmp(&var[a]));
    dims.truncate(3);
    let range: Vec<(f32, f32)> = dims
        .iter()
        .map(|&t| (0..n).map(|i| x.row(i)[t]).fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v))))
        .collect();
    let code = |i: usize| -> u32 {
        let mut c = 0u32;
        for (slot, (&t, &(lo, hi))) in dims.iter().zip(&range).enumerate() {
            let q = if hi > lo { (((x.row(i)[t] - lo) / (hi - lo)) * 1023.0) as u32 } else { 0 };
            for b in 0..10 {
                c |= ((q >> b) & 1) << (3 * b + slot);
            }
        }
        c
    };
    let mut keyed: Vec<(u32, u32)> = (0..n).into_par_iter().map(|i| (code(i), i as u32)).collect();
    keyed.par_sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// The kernel's `cap` candidates (positions in the device order) for every row, in blocks
/// of `qb` queries.
fn candidates(gpu: &Arc<Gpu>, xc: &[f32], norms: &[f32], n: usize, d: usize, cap: usize, qb: usize) -> Result<Vec<u32>> {
    let (kernel, tp) = gpu.knn_kernel(d, cap)?;
    let group = KNN_BLOCK as usize * KNN_QPT;
    // Launch boundaries must fall on query-box boundaries.
    let qb = if qb >= n { n } else { (qb / group).max(1) * group };
    guard("CUDA kNN", || {
        let xc_d = gpu.upload(xc, "upload embeddings")?;
        let norms_d = gpu.upload(norms, "upload norms")?;
        let qbox_d = gpu.upload(&boxes(xc, d, group), "upload query boxes")?;
        let tbox_d = gpu.upload(&boxes(xc, d, tp), "upload tile boxes")?;
        let mut out_idx = gpu.alloc_zeros::<u32>(qb * cap, "alloc candidate indices")?;
        let mut cand = vec![0u32; n * cap];

        let mut q0 = 0usize;
        while q0 < n {
            let nq = qb.min(n - q0);
            let (ni, q0i, nqi) = (n as i32, q0 as i32, nq as i32);
            let cfg = LaunchConfig {
                grid_dim: (nq.div_ceil(KNN_BLOCK as usize * KNN_QPT) as u32, 1, 1),
                block_dim: (KNN_BLOCK, 1, 1),
                shared_mem_bytes: (4 * tp * (d + 1)) as u32,
            };
            // SAFETY: argument order and types match the `knn` kernel compiled for this
            // (d, cap); the shared tile holds tp * (d + 1) floats.
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&xc_d)
                    .arg(&norms_d)
                    .arg(&qbox_d)
                    .arg(&tbox_d)
                    .arg(&ni)
                    .arg(&q0i)
                    .arg(&nqi)
                    .arg(&mut out_idx)
                    .launch(cfg)
            }
            .map_err(|e| drv_err("launch knn", e))?;
            let dst = &mut cand[q0 * cap..(q0 + nq) * cap];
            gpu.stream.memcpy_dtoh(&out_idx.slice(..nq * cap), dst).map_err(|e| drv_err("download candidates", e))?;
            gpu.sync("download candidates")?;
            q0 += nq;
        }
        Ok(cand)
    })
}

