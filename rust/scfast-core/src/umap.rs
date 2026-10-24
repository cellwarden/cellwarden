//! UMAP layout from a kNN graph, replacing `RunUMAP` (uwot) and `sc.tl.umap` (umap-learn).
//!
//! Same algorithm as umap-learn: fuzzy simplicial set from the kNN distances
//! (`smooth_knn_dist` + `compute_membership_strengths`), fuzzy union, then the SGD layout
//! (`optimize_layout_euclidean`) with edge sampling by weight and negative sampling. The SGD
//! runs edges in parallel without locks (Hogwild, as umap-learn's `parallel=True`), so the
//! layout is not bit-reproducible across thread counts; the embedding coordinates live in
//! relaxed atomics so concurrent updates are defined behaviour.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use rayon::prelude::*;

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct UmapParams {
    /// Curve parameters of the low-dimensional similarity `1 / (1 + a d^(2b))`.
    pub a: f32,
    pub b: f32,
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

/// Fuzzy union of the kNN membership strengths, as a symmetric CSR graph without self loops.
pub struct FuzzyGraph {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;
/// Edges per SGD task at least: every epoch ends in a barrier, so small graphs stay on few threads.
const MIN_TASK_EDGES: usize = 16_384;

/// `indices` / `distances` are (n, k) row-major with the cell itself in column 0.
pub fn fuzzy_simplicial_set(indices: &[u32], distances: &[f32], n: usize, k: usize) -> Result<FuzzyGraph> {
    if indices.len() != n * k || distances.len() != n * k || k < 2 || n * k > u32::MAX as usize {
        return Err(Error::InvalidInput(format!("kNN arrays must be {n} x {k} with k >= 2 and n k < 2^32")));
    }
    let mean_all = distances.iter().map(|&d| d as f64).sum::<f64>() / distances.len() as f64;
    let target = (k as f32).log2();
    let mut w = vec![0f32; n * k];
    w.par_chunks_mut(k).enumerate().for_each(|(i, wi)| {
        let d = &distances[i * k..(i + 1) * k];
        let rho = d.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0);
        let (mut lo, mut hi, mut mid) = (0f32, f32::INFINITY, 1f32);
        for _ in 0..64 {
            let psum: f32 = d[1..].iter().map(|&x| {
                let e = x - rho;
                if e > 0.0 { (-e / mid).exp() } else { 1.0 }
            }).sum();
            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                mid = if hi.is_infinite() { mid * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        let mean_i = d.iter().sum::<f32>() / k as f32;
        let floor = MIN_K_DIST_SCALE * if rho > 0.0 { mean_i } else { mean_all as f32 };
        let sigma = mid.max(floor);
        for j in 0..k {
            let nb = indices[i * k + j] as usize;
            wi[j] = if nb == i {
                0.0
            } else if d[j] - rho <= 0.0 {
                1.0
            } else {
                (-(d[j] - rho) / sigma).exp()
            };
        }
    });

    // Row i of the union holds its own kNN edges and the reversed edges j -> i, gathered
    // through a transpose filled in parallel; blocks of rows merge into a few large vectors.
    let incoming: Vec<AtomicU32> = (0..=n).map(|_| AtomicU32::new(0)).collect();
    indices.par_iter().zip(w.par_iter()).for_each(|(&j, &wp)| {
        if wp > 0.0 {
            incoming[j as usize + 1].fetch_add(1, Relaxed);
        }
    });
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        start[j + 1] = start[j] + incoming[j + 1].load(Relaxed) as usize;
    }
    let cursor: Vec<AtomicU32> = start[..n].iter().map(|&s| AtomicU32::new(s as u32)).collect();
    let src: Vec<AtomicU32> = (0..start[n]).map(|_| AtomicU32::new(0)).collect();
    w.par_chunks(k).enumerate().for_each(|(i, wi)| {
        for (q, &wp) in wi.iter().enumerate() {
            if wp > 0.0 {
                let pos = cursor[indices[i * k + q] as usize].fetch_add(1, Relaxed);
                src[pos as usize].store((i * k + q) as u32, Relaxed);
            }
        }
    });

    const ROWS: usize = 4096;
    let blocks: Vec<(Vec<usize>, Vec<u32>, Vec<f32>)> = (0..n.div_ceil(ROWS)).into_par_iter().map(|bl| {
        let (mut lens, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let mut r: Vec<(u32, f32)> = Vec::new();
        for i in bl * ROWS..((bl + 1) * ROWS).min(n) {
            r.clear();
            r.extend((i * k..(i + 1) * k).filter(|&p| w[p] > 0.0).map(|p| (indices[p], w[p])));
            r.extend(src[start[i]..start[i + 1]].iter().map(|p| {
                let p = p.load(Relaxed) as usize;
                ((p / k) as u32, w[p])
            }));
            r.sort_unstable_by_key(|e| e.0);
            let before = cols.len();
            for &(c, v) in &r {
                if cols.len() > before && *cols.last().unwrap() == c {
                    let last: &mut f32 = vals.last_mut().unwrap();
                    *last = *last + v - *last * v;
                } else {
                    cols.push(c);
                    vals.push(v);
                }
            }
            lens.push(cols.len() - before);
        }
        (lens, cols, vals)
    }).collect();

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for (lens, _, _) in &blocks {
        for l in lens {
            indptr.push(indptr.last().unwrap() + l);
        }
    }
    let mut indices_out = Vec::with_capacity(indptr[n]);
    let mut weights = Vec::with_capacity(indptr[n]);
    for (_, cols, vals) in &blocks {
        indices_out.extend_from_slice(cols);
        weights.extend_from_slice(vals);
    }
    Ok(FuzzyGraph { indptr, indices: indices_out, weights })
}

#[inline(always)]
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Interleaves the bits of two 16-bit coordinates (Z-order curve).
fn morton(x: u32, y: u32) -> u32 {
    let spread = |mut v: u32| {
        v &= 0xFFFF;
        v = (v | (v << 8)) & 0x00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333;
        (v | (v << 1)) & 0x5555_5555
    };
    spread(x) | (spread(y) << 1)
}

/// Hints the cache to load the line holding `p`; negative samples are drawn ahead and
/// prefetched, so their random reads over the embedding overlap.
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
}

/// Negative samples drawn this many draws before they are used.
const NEG_AHEAD: usize = 16;

#[inline(always)]
fn clip(g: f32) -> f32 {
    g.clamp(-4.0, 4.0)
}

const LANES: usize = 32;

/// Pending samples of one head cell: the other cell's position and the weight of its step (2
/// for a graph edge, whose attraction is doubled for head-only updates; 1 for a negative
/// sample, which repels; 0 for an empty lane). The count of used lanes is kept by the caller,
/// in a register.
#[derive(Default)]
struct Lanes {
    x: [f32; LANES],
    y: [f32; LANES],
    w: [f32; LANES],
}

impl Lanes {
    #[inline(always)]
    fn set(&mut self, l: usize, o: [f32; 2], w: f32) {
        (self.x[l], self.y[l], self.w[l]) = (o[0], o[1], w);
    }

    /// The summed clipped gradient steps at `cur` of the first `m` lanes (umap-learn's formulas).
    #[inline(always)]
    fn forces(&mut self, m: usize, cur: [f32; 2], a: f32, b: f32, avx2: bool) -> [f32; 2] {
        for l in m..m.next_multiple_of(8) {
            self.set(l, [0.0, 0.0], 0.0);
        }
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: avx2 is true only when use_avx2() detected AVX2 and FMA.
            return unsafe { forces_avx2(self, m, cur, a, b) };
        }
        let _ = avx2;
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { forces_neon(self, m, cur, a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.forces_scalar(m, cur, a, b)
    }

    fn forces_scalar(&self, m: usize, cur: [f32; 2], a: f32, b: f32) -> [f32; 2] {
        let mut f = [0f32; 2];
        for q in 0..m {
            let dx = [cur[0] - self.x[q], cur[1] - self.y[q]];
            let d2 = dx[0] * dx[0] + dx[1] * dx[1];
            let w = self.w[q];
            for d in 0..2 {
                f[d] += w * if w > 1.5 {
                    if d2 > 0.0 {
                        let pb = d2.powf(b);
                        clip(-2.0 * a * b * pb / (d2 * (a * pb + 1.0)) * dx[d])
                    } else {
                        0.0
                    }
                } else if d2 > 0.0 {
                    clip(2.0 * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0)) * dx[d])
                } else {
                    4.0
                };
            }
        }
        f
    }
}

#[inline]
fn use_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `Lanes::forces_scalar` 8 lanes at a time (lanes up to the next multiple of 8 must have
/// weight 0), with `d2^b` as `exp2(b log2 d2)` from polynomials (relative error below 1e-5)
/// instead of a libm `powf` per sample. The groups are independent, so their long dependency
/// chains overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn forces_avx2(l: &Lanes, m: usize, cur: [f32; 2], a: f32, b: f32) -> [f32; 2] {
    use std::arch::x86_64::*;
    let c = _mm256_set1_ps;
    let ci = _mm256_set1_epi32;
    let (mut sx, mut sy) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for o in (0..m).step_by(8) {
        let (x, y, w) = (_mm256_loadu_ps(l.x[o..].as_ptr()), _mm256_loadu_ps(l.y[o..].as_ptr()), _mm256_loadu_ps(l.w[o..].as_ptr()));
        let (dx, dy) = (_mm256_sub_ps(c(cur[0]), x), _mm256_sub_ps(c(cur[1]), y));
        let d2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
        let d2c = _mm256_max_ps(d2, c(f32::MIN_POSITIVE));

        // log2(d2c) = e + log2(r), r in [sqrt(1/2), sqrt(2)), ln r = 2 atanh((r - 1) / (r + 1)).
        let bits = _mm256_castps_si256(d2c);
        let mant = _mm256_and_si256(bits, ci(0x007F_FFFF));
        let big = _mm256_cmpgt_epi32(mant, ci(0x0035_04F3));
        let e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_sub_epi32(_mm256_srli_epi32(bits, 23), ci(127)), big));
        let r = _mm256_castsi256_ps(_mm256_or_si256(mant, _mm256_blendv_epi8(ci(0x3F80_0000), ci(0x3F00_0000), big)));
        let t = _mm256_div_ps(_mm256_sub_ps(r, c(1.0)), _mm256_add_ps(r, c(1.0)));
        let t2 = _mm256_mul_ps(t, t);
        let mut s = _mm256_fmadd_ps(t2, c(1.0 / 7.0), c(1.0 / 5.0));
        s = _mm256_fmadd_ps(t2, s, c(1.0 / 3.0));
        s = _mm256_fmadd_ps(t2, s, c(1.0));
        let ln_r = _mm256_mul_ps(_mm256_mul_ps(c(2.0), t), s);
        let z = _mm256_mul_ps(c(b), _mm256_fmadd_ps(ln_r, c(std::f32::consts::LOG2_E), e));
        let z = _mm256_min_ps(_mm256_max_ps(z, c(-126.0)), c(126.0));

        // 2^z = 2^i e^(f ln 2), i = round(z), |f| <= 1/2.
        let i = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(z);
        let f = _mm256_mul_ps(_mm256_sub_ps(z, i), c(std::f32::consts::LN_2));
        let mut p = _mm256_fmadd_ps(f, c(1.0 / 720.0), c(1.0 / 120.0));
        for k in [1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
            p = _mm256_fmadd_ps(f, p, c(k));
        }
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(i), ci(127)), 23));
        let pb = _mm256_mul_ps(p, scale);

        let att = _mm256_cmp_ps::<_CMP_GT_OQ>(w, c(1.5));
        let num = _mm256_blendv_ps(c(2.0 * b), _mm256_mul_ps(c(-2.0 * a * b), pb), att);
        let den = _mm256_mul_ps(_mm256_blendv_ps(_mm256_add_ps(d2, c(0.001)), d2c, att), _mm256_fmadd_ps(c(a), pb, c(1.0)));
        let g = _mm256_div_ps(num, den);
        let clip8 = |v: __m256| _mm256_min_ps(_mm256_max_ps(v, c(-4.0)), c(4.0));
        let zero = _mm256_blendv_ps(c(4.0), c(0.0), att);
        let live = _mm256_cmp_ps::<_CMP_GT_OQ>(d2, c(0.0));
        let fx = _mm256_mul_ps(w, _mm256_blendv_ps(zero, clip8(_mm256_mul_ps(g, dx)), live));
        let fy = _mm256_mul_ps(w, _mm256_blendv_ps(zero, clip8(_mm256_mul_ps(g, dy)), live));
        sx = _mm256_add_ps(sx, fx);
        sy = _mm256_add_ps(sy, fy);
    }
    let (mut ox, mut oy) = ([0f32; 8], [0f32; 8]);
    _mm256_storeu_ps(ox.as_mut_ptr(), sx);
    _mm256_storeu_ps(oy.as_mut_ptr(), sy);
    [ox.iter().sum(), oy.iter().sum()]
}

/// [`forces_avx2`] on aarch64 NEON, 4 lanes per vector and two vectors per step.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn forces_neon(l: &Lanes, m: usize, cur: [f32; 2], a: f32, b: f32) -> [f32; 2] {
    use std::arch::aarch64::*;
    let c = vdupq_n_f32;
    let ci = vdupq_n_u32;
    let (mut sx, mut sy) = (c(0.0), c(0.0));
    for o in (0..m).step_by(4) {
        let (x, y, w) = (vld1q_f32(l.x[o..].as_ptr()), vld1q_f32(l.y[o..].as_ptr()), vld1q_f32(l.w[o..].as_ptr()));
        let (dx, dy) = (vsubq_f32(c(cur[0]), x), vsubq_f32(c(cur[1]), y));
        let d2 = vfmaq_f32(vmulq_f32(dy, dy), dx, dx);
        let d2c = vmaxq_f32(d2, c(f32::MIN_POSITIVE));

        let bits = vreinterpretq_u32_f32(d2c);
        let mant = vandq_u32(bits, ci(0x007F_FFFF));
        let big = vcgtq_u32(mant, ci(0x0035_04F3));
        let e = vcvtq_f32_s32(vsubq_s32(vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32::<23>(bits)), vdupq_n_s32(127)), vreinterpretq_s32_u32(big)));
        let r = vreinterpretq_f32_u32(vorrq_u32(mant, vbslq_u32(big, ci(0x3F00_0000), ci(0x3F80_0000))));
        let t = vdivq_f32(vsubq_f32(r, c(1.0)), vaddq_f32(r, c(1.0)));
        let t2 = vmulq_f32(t, t);
        let mut s = vfmaq_f32(c(1.0 / 5.0), t2, c(1.0 / 7.0));
        s = vfmaq_f32(c(1.0 / 3.0), t2, s);
        s = vfmaq_f32(c(1.0), t2, s);
        let ln_r = vmulq_f32(vmulq_f32(c(2.0), t), s);
        let z = vmulq_f32(c(b), vfmaq_f32(e, ln_r, c(std::f32::consts::LOG2_E)));
        let z = vminq_f32(vmaxq_f32(z, c(-126.0)), c(126.0));

        let i = vrndnq_f32(z);
        let f = vmulq_f32(vsubq_f32(z, i), c(std::f32::consts::LN_2));
        let mut p = vfmaq_f32(c(1.0 / 120.0), f, c(1.0 / 720.0));
        for k in [1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
            p = vfmaq_f32(c(k), f, p);
        }
        let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(i), vdupq_n_s32(127))));
        let pb = vmulq_f32(p, scale);

        let att = vcgtq_f32(w, c(1.5));
        let num = vbslq_f32(att, vmulq_f32(c(-2.0 * a * b), pb), c(2.0 * b));
        let den = vmulq_f32(vbslq_f32(att, d2c, vaddq_f32(d2, c(0.001))), vfmaq_f32(c(1.0), c(a), pb));
        let g = vdivq_f32(num, den);
        let clip4 = |v: float32x4_t| vminq_f32(vmaxq_f32(v, c(-4.0)), c(4.0));
        let zero = vbslq_f32(att, c(0.0), c(4.0));
        let live = vcgtq_f32(d2, c(0.0));
        sx = vfmaq_f32(sx, w, vbslq_f32(live, clip4(vmulq_f32(g, dx)), zero));
        sy = vfmaq_f32(sy, w, vbslq_f32(live, clip4(vmulq_f32(g, dy)), zero));
    }
    [vaddvq_f32(sx), vaddvq_f32(sy)]
}

/// SGD layout of `graph` starting from `init` (n x 2, row-major). Returns the n x 2 embedding.
pub fn optimize_layout(graph: &FuzzyGraph, init: &[f32], p: &UmapParams) -> Result<Vec<f32>> {
    const DIM: usize = 2;
    let n = graph.indptr.len() - 1;
    if init.len() != n * DIM {
        return Err(Error::InvalidInput(format!("init must be {n} x {DIM}")));
    }
    if p.n_epochs == 0 {
        return Err(Error::InvalidParameter("n_epochs must be at least 1".into()));
    }
    let (mut lo, mut hi) = ([f32::INFINITY; DIM], [f32::NEG_INFINITY; DIM]);
    for r in init.chunks_exact(DIM) {
        for d in 0..DIM {
            lo[d] = lo[d].min(r[d]);
            hi[d] = hi[d].max(r[d]);
        }
    }
    let span: Vec<f32> = (0..DIM).map(|d| if hi[d] > lo[d] { hi[d] - lo[d] } else { 1.0 }).collect();
    // Cells are relabelled in Z order of their start positions, so graph neighbours sit close
    // in memory and each thread's run of heads covers one region of the layout.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let code = |i: usize| {
        let q = |d: usize| ((init[i * DIM + d] - lo[d]) / span[d] * 65535.0) as u32;
        morton(q(0), q(1))
    };
    let codes: Vec<u32> = (0..n).into_par_iter().map(code).collect();
    order.par_sort_unstable_by_key(|&i| (codes[i as usize], i));
    let mut rank = vec![0u32; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }

    // Edge e is sampled at the epochs where floor(epoch / eps_e) steps up, with eps_e =
    // wmax / w_e, and draws `negative_sample_rate` negatives each time: umap-learn's schedule
    // without its per-edge counters, so an epoch reads 8 bytes per edge and writes none.
    // Heads are implicit in the row offsets of the relabelled graph.
    let wmax = graph.weights.iter().copied().fold(0f32, f32::max);
    let cutoff = wmax / p.n_epochs as f32;
    let kept = |i: u32| {
        let i = i as usize;
        (graph.indptr[i]..graph.indptr[i + 1]).filter(|&q| graph.weights[q] >= cutoff && graph.weights[q] > 0.0)
    };
    let mut offsets = vec![0usize; n + 1];
    let counts: Vec<usize> = order.par_iter().map(|&i| kept(i).count()).collect();
    for (r, c) in counts.into_iter().enumerate() {
        offsets[r + 1] = offsets[r] + c;
    }
    // (tail, 1 / eps) per edge, rows in relabelled order.
    let edges: Vec<(u32, f32)> = order.par_iter().flat_map_iter(|&i| {
        kept(i).map(|q| (rank[graph.indices[q] as usize], graph.weights[q] / wmax))
    }).collect();
    // Tasks are runs of whole rows of about CHUNK edges, so each head is written by one task.
    const CHUNK: usize = 4096;
    let mut tasks = vec![0usize];
    for r in 1..=n {
        if offsets[r] - offsets[*tasks.last().unwrap()] >= CHUNK || r == n {
            tasks.push(r);
        }
    }
    let tasks: Vec<(usize, usize)> = tasks.windows(2).map(|w| (w[0], w[1])).collect();

    let emb: Vec<AtomicU32> = order.iter().flat_map(|&i| {
        let span = &span;
        (0..DIM).map(move |d| AtomicU32::new((10.0 * (init[i as usize * DIM + d] - lo[d]) / span[d]).to_bits()))
    }).collect();

    let (a, b) = (p.a, p.b);
    let n_neg = p.negative_sample_rate;
    let avx2 = use_avx2();
    let get = |i: usize| [f32::from_bits(emb[i * DIM].load(Relaxed)), f32::from_bits(emb[i * DIM + 1].load(Relaxed))];
    let put = |i: usize, v: [f32; DIM]| {
        emb[i * DIM].store(v[0].to_bits(), Relaxed);
        emb[i * DIM + 1].store(v[1].to_bits(), Relaxed);
    };

    for epoch in 1..p.n_epochs {
        let (ep, prev) = (epoch as f32, (epoch - 1) as f32);
        let alpha = p.learning_rate * (1.0 - ep / p.n_epochs as f32);
        tasks.par_iter().enumerate().with_min_len(MIN_TASK_EDGES / CHUNK).for_each(|(c, &(r0, r1))| {
            let seed = (p.seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(epoch as u64 + 1).wrapping_add((c as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
            // One xorshift stream per ring slot, so consecutive draws do not wait on each other.
            let mut rngs: [u64; NEG_AHEAD] = std::array::from_fn(|s| seed.wrapping_add((s as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)) | 1);
            let draw = |rng: &mut u64| {
                let q = (((xorshift(rng) >> 32) * n as u64) >> 32) as usize;
                prefetch(&emb[q * DIM]);
                q
            };
            let mut ahead: [usize; NEG_AHEAD] = std::array::from_fn(|s| draw(&mut rngs[s]));
            let mut slot = 0;
            let mut lanes = Lanes::default();
            for j in r0..r1 {
                let (cur, mut delta, mut m) = (get(j), [0f32; DIM], 0);
                let mut add = |lanes: &mut Lanes, m: &mut usize| {
                    let f = lanes.forces(*m, cur, a, b, avx2);
                    delta[0] += f[0];
                    delta[1] += f[1];
                    *m = 0;
                };
                // The due tails of each run of up to 64 edges are packed without branches
                // (whether an edge is due is close to random), then sampled.
                for run in edges[offsets[j]..offsets[j + 1]].chunks(64) {
                    let (mut due, mut k) = ([0u32; 64], 0);
                    for &(tail, ie) in run {
                        due[k] = tail;
                        k += ((ep * ie).floor() > (prev * ie).floor()) as usize;
                    }
                    for &tail in &due[..k] {
                        lanes.set(m, get(tail as usize), 2.0);
                        m += 1;
                        if m == LANES {
                            add(&mut lanes, &mut m);
                        }
                    }
                    for _ in 0..k * n_neg {
                        let q = std::mem::replace(&mut ahead[slot], draw(&mut rngs[slot]));
                        slot = (slot + 1) % NEG_AHEAD;
                        if q != j {
                            lanes.set(m, get(q), 1.0);
                            m += 1;
                            if m == LANES {
                                add(&mut lanes, &mut m);
                            }
                        }
                    }
                }
                if m > 0 {
                    add(&mut lanes, &mut m);
                }
                if delta != [0.0; DIM] {
                    put(j, [cur[0] + alpha * delta[0], cur[1] + alpha * delta[1]]);
                }
            }
        });
    }
    let mut out = vec![0f32; n * DIM];
    for (r, &i) in order.iter().enumerate() {
        for d in 0..DIM {
            out[i as usize * DIM + d] = f32::from_bits(emb[r * DIM + d].load(Relaxed));
        }
    }
    Ok(out)
}

/// Fuzzy simplicial set + SGD layout in one call.
pub fn umap(indices: &[u32], distances: &[f32], n: usize, k: usize, init: &[f32], p: &UmapParams) -> Result<Vec<f32>> {
    let g = fuzzy_simplicial_set(indices, distances, n, k)?;
    optimize_layout(&g, init, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_forces_match_scalar() {
        if !use_avx2() {
            return;
        }
        let (a, b) = (0.583, 1.334);
        let mut s = 11u64;
        let mut u = || (xorshift(&mut s) % 100_000) as f32 / 5000.0;
        for m in 1..=LANES {
            for _ in 0..500 {
                let mut l = Lanes::default();
                let cur = [u(), u()];
                for q in 0..m {
                    // Some samples sit on the head itself (d2 = 0) or very close to it.
                    let o = match q % 4 { 0 => cur, 1 => [cur[0] + 1e-4, cur[1]], _ => [u(), u()] };
                    l.set(q, o, if q % 3 == 0 { 2.0 } else { 1.0 });
                }
                let want = l.forces_scalar(m, cur, a, b);
                let got = unsafe { forces_avx2(&l, m, cur, a, b) };
                for d in 0..2 {
                    assert!((got[d] - want[d]).abs() <= 1e-4 * (1.0 + want[d].abs()), "{got:?} vs {want:?}");
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_forces_match_scalar() {
        let (a, b) = (0.583, 1.334);
        let mut s = 11u64;
        let mut u = || (xorshift(&mut s) % 100_000) as f32 / 5000.0;
        for m in 1..=LANES {
            for _ in 0..500 {
                let mut l = Lanes::default();
                let cur = [u(), u()];
                for q in 0..m {
                    let o = match q % 4 { 0 => cur, 1 => [cur[0] + 1e-4, cur[1]], _ => [u(), u()] };
                    l.set(q, o, if q % 3 == 0 { 2.0 } else { 1.0 });
                }
                let want = l.forces_scalar(m, cur, a, b);
                let got = unsafe { forces_neon(&l, m, cur, a, b) };
                for d in 0..2 {
                    assert!((got[d] - want[d]).abs() <= 1e-4 * (1.0 + want[d].abs()), "{got:?} vs {want:?}");
                }
            }
        }
    }

    #[test]
    fn two_blobs_separate() {
        let (n, k) = (400usize, 10usize);
        let mut s = 7u64;
        let pts: Vec<[f32; 2]> = (0..n).map(|i| {
            let c = if i < n / 2 { 0.0 } else { 50.0 };
            [c + (xorshift(&mut s) % 1000) as f32 / 100.0, (xorshift(&mut s) % 1000) as f32 / 100.0]
        }).collect();
        let mut idx = vec![0u32; n * k];
        let mut dist = vec![0f32; n * k];
        for i in 0..n {
            let mut o: Vec<(f32, u32)> = (0..n).map(|j| {
                let d = ((pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2)).sqrt();
                (if i == j { -1.0 } else { d }, j as u32)
            }).collect();
            o.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for q in 0..k {
                idx[i * k + q] = o[q].1;
                dist[i * k + q] = o[q].0.max(0.0);
            }
        }
        let g = fuzzy_simplicial_set(&idx, &dist, n, k).unwrap();
        for i in 0..n {
            for q in g.indptr[i]..g.indptr[i + 1] {
                let j = g.indices[q] as usize;
                let back = (g.indptr[j]..g.indptr[j + 1]).find(|&r| g.indices[r] as usize == i).unwrap();
                assert!((g.weights[q] - g.weights[back]).abs() < 1e-6);
            }
        }
        let init: Vec<f32> = pts.iter().flat_map(|p| [p[1], p[0]]).collect();
        for seed in 0..3 {
            let p = UmapParams { a: 0.583, b: 1.334, n_epochs: 200, negative_sample_rate: 5, learning_rate: 1.0, seed };
            let e = optimize_layout(&g, &init, &p).unwrap();
            assert!(e.iter().all(|v| v.is_finite()));
            // Each cell's 10 nearest cells in the layout come from its own blob. (A blob may
            // stretch or split, so distance to the blob centres is not a stable criterion.)
            let d = |i: usize, j: usize| (e[2 * i] - e[2 * j]).powi(2) + (e[2 * i + 1] - e[2 * j + 1]).powi(2);
            let mixed = (0..n).filter(|&i| {
                let mut o: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                o.sort_by(|&x, &y| d(i, x).total_cmp(&d(i, y)));
                o[..10].iter().any(|&j| (j < n / 2) != (i < n / 2))
            }).count();
            assert_eq!(mixed, 0, "seed {seed}: {mixed} cells have a neighbour from the other blob");
        }
    }
}
