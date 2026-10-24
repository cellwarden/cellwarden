//! The implicitly scaled operator on a CUDA device (feature `cuda`).
//!
//! The CSR of the HVG matrix is uploaded once (i64 row offsets, u32 column indices, f32
//! values) with the per-gene mean and 1/sd. Each Krylov step then does:
//!
//! - `apply`:   host builds `vs = diag(1/sd) V` (row-major f32) and `shift = mu^T vs` (f64,
//!   rounded once), uploads them, runs the `zv` kernel, downloads the cells x l result.
//! - `apply_t`: host uploads `U` (f32, column-major) and its column sums (f64 on the host),
//!   zeroes the genes x l accumulator, runs `ztu` (atomic scatter) and `ztu_finish` (mean
//!   and 1/sd correction), downloads the genes x l result.
//!
//! - `gram_top_eigen`: Z^T Z in one pass (`xtx`, `gram_finish`) and its eigendecomposition
//!   (cuSOLVER `syevd`), for the Gram path in `pca.rs`; `gram` downloads Z^T Z instead.
//! - `CsrDev::cap`: Seurat's clip, as a per-gene cap on the stored values (`cap` kernel).
//!
//! All device buffers (but the Gram matrix) are allocated at construction for the fixed `l`, so the working set
//! is known up front and checked against free memory; an allocation that still fails is
//! reported as `Error::Gpu("out of device memory ...")`.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtrMut, DeviceRepr, PushKernelArg};
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};

use super::operator::StatsAcc;
use crate::cuda::{drv_err, guard, Gpu};
use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real};

/// Elements per host-to-device staging chunk for the CSR upload (16 MiB of f32).
const STAGE: usize = 4 << 20;

pub(super) struct CudaScaledOp {
    gpu: Arc<Gpu>,
    n: usize,
    p: usize,
    l: usize,
    indptr: CudaSlice<i64>,
    indices: CudaSlice<u32>,
    vals: CudaSlice<f32>,
    mean_d: CudaSlice<f32>,
    inv_sd_d: CudaSlice<f32>,
    mean: Vec<f64>,
    inv_sd: Vec<f64>,
    /// Column sums of the stored X (equal to n mean unless the values were capped).
    x_colsum: Vec<f64>,
    // Scratch, reused by every product (apply takes &self, hence RefCell).
    vs: RefCell<CudaSlice<f32>>,
    shift: RefCell<CudaSlice<f32>>,
    colsum: RefCell<CudaSlice<f32>>,
    dense_n: RefCell<CudaSlice<f32>>,
    acc: RefCell<CudaSlice<f32>>,
    out_p: RefCell<CudaSlice<f32>>,
    host_n: RefCell<Vec<f32>>,
    host_p: RefCell<Vec<f32>>,
}

/// A CSR matrix on the device in the layout the kernels read: i64 row offsets, u32 column
/// indices, f32 values.
pub(crate) struct CsrDev {
    gpu: Arc<Gpu>,
    pub n: usize,
    pub p: usize,
    indptr: CudaSlice<i64>,
    indices: CudaSlice<u32>,
    vals: CudaSlice<f32>,
}

/// Borrowed CSR arrays already on the device (e.g. a CuPy matrix): `n + 1` row offsets
/// (int32 or int64), `nnz` int32 column indices and `nnz` f32 values, canonical (sorted, no
/// duplicates) with indices below `p`. Nothing of this is checked on the device.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCsr {
    pub n: usize,
    pub p: usize,
    pub nnz: usize,
    pub indptr: u64,
    pub indptr_i64: bool,
    pub indices: u64,
    pub data: u64,
}

fn check_dims(n: usize, p: usize) -> Result<()> {
    if n > i32::MAX as usize || p > i32::MAX as usize {
        return Err(Error::gpu(format!("{n} cells x {p} genes exceeds the CUDA kernel index range")));
    }
    Ok(())
}

impl CsrDev {
    /// Uploads a host CSR through a bounded staging buffer.
    pub fn upload<I: Idx, T: Real>(x: &CsrView<'_, I, T>, ordinal: usize) -> Result<Self> {
        let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
        check_dims(n, p)?;
        let gpu = Gpu::get(ordinal)?;
        gpu.ensure_fits(8 * (n + 1) + 8 * nnz, &format!("CSR of {n} cells x {p} genes ({nnz} non-zeros)"))?;
        guard("CUDA CSR upload", || {
            let indptr = upload_chunked(&gpu, n + 1, |r| x.indptr[r.clone()].iter().map(|v| v.to_usize() as i64).collect(), "upload indptr")?;
            let indices = upload_chunked(&gpu, nnz, |r| x.indices[r.clone()].iter().map(|v| v.to_usize() as u32).collect(), "upload indices")?;
            let vals = upload_chunked(&gpu, nnz, |r| x.data[r.clone()].iter().map(|v| v.to_f32()).collect(), "upload values")?;
            Ok(Self { gpu: gpu.clone(), n, p, indptr, indices, vals })
        })
    }

    /// Copies a CSR that is already on the device (device-to-device; int32 row offsets are
    /// widened through the host, which is n + 1 integers).
    ///
    /// # Safety
    /// The pointers in `x` must be valid device memory of the stated sizes and types on
    /// device `ordinal`, with all work writing them finished.
    pub unsafe fn copy_from(x: &DeviceCsr, ordinal: usize) -> Result<Self> {
        let (n, p, nnz) = (x.n, x.p, x.nnz);
        check_dims(n, p)?;
        let gpu = Gpu::get(ordinal)?;
        gpu.ensure_fits(8 * (n + 1) + 8 * nnz, &format!("CSR of {n} cells x {p} genes ({nnz} non-zeros)"))?;
        guard("CUDA CSR copy", || {
            let stream = &gpu.stream;
            let dtod = |dst: cudarc::driver::sys::CUdeviceptr, src: u64, bytes: usize, what: &str| -> Result<()> {
                // SAFETY: both ranges are `bytes` long on this device (caller's contract).
                unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }.map_err(|e| drv_err(what, e))
            };
            let mut indices = unsafe { stream.alloc::<u32>(nnz) }.map_err(|e| drv_err("alloc indices", e))?;
            let mut vals = unsafe { stream.alloc::<f32>(nnz) }.map_err(|e| drv_err("alloc values", e))?;
            {
                let (d, _w) = indices.device_ptr_mut(stream);
                dtod(d, x.indices, 4 * nnz, "copy indices")?;
            }
            {
                let (d, _w) = vals.device_ptr_mut(stream);
                dtod(d, x.data, 4 * nnz, "copy values")?;
            }
            let indptr = if x.indptr_i64 {
                let mut ip = unsafe { stream.alloc::<i64>(n + 1) }.map_err(|e| drv_err("alloc indptr", e))?;
                {
                    let (d, _w) = ip.device_ptr_mut(stream);
                    dtod(d, x.indptr, 8 * (n + 1), "copy indptr")?;
                }
                ip
            } else {
                let mut host = vec![0i32; n + 1];
                // SAFETY: n + 1 int32 values at x.indptr (caller's contract).
                unsafe { cudarc::driver::result::memcpy_dtoh_async(&mut host, x.indptr, stream.cu_stream()) }.map_err(|e| drv_err("read indptr", e))?;
                gpu.sync("read indptr")?;
                gpu.upload(&host.iter().map(|&v| v as i64).collect::<Vec<i64>>(), "upload indptr")?
            };
            gpu.sync("copy CSR")?;
            Ok(Self { gpu: gpu.clone(), n, p, indptr, indices, vals })
        })
    }

    /// Caps every stored value at `caps[column]` (`cap` kernel, in place).
    pub fn cap(&mut self, caps: &[f32]) -> Result<()> {
        let nnz = self.vals.len() as u64;
        guard("CUDA clip", || {
            let caps_d = self.gpu.upload(caps, "upload caps")?;
            let cfg = cudarc::driver::LaunchConfig { grid_dim: (4096, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: argument types and order match the `cap` kernel; `caps` has p entries.
            unsafe { self.gpu.stream.launch_builder(&self.gpu.cap).arg(&self.indices).arg(&mut self.vals).arg(&caps_d).arg(&nnz).launch(cfg) }
                .map_err(|e| drv_err("launch cap", e))?;
            self.gpu.sync("cap")
        })
    }

    /// Per-gene accumulators of `operator::gene_stats`, from the `colstats` kernel.
    pub fn stats(&self) -> Result<StatsAcc> {
        const GCH: usize = 1536; // as in the kernel
        let (n, p) = (self.n, self.p);
        let (mut sum, mut sumsq) = (vec![0f64; p], vec![0f64; p]);
        let (mut mn, mut mx) = (vec![f32::INFINITY; p], vec![f32::NEG_INFINITY; p]);
        let mut cnt = vec![0u32; p];
        guard("CUDA gene statistics", || {
            let gpu = &self.gpu;
            let mut sum_d = gpu.alloc_zeros::<f64>(p, "alloc sums")?;
            let mut sumsq_d = gpu.alloc_zeros::<f64>(p, "alloc sums of squares")?;
            let mut mn_d = gpu.upload(&mn, "init minima")?;
            let mut mx_d = gpu.upload(&mx, "init maxima")?;
            let mut cnt_d = gpu.alloc_zeros::<u32>(p, "alloc counts")?;
            let (ni, pi) = (n as i32, p as i32);
            let cfg = cudarc::driver::LaunchConfig { grid_dim: (512, p.div_ceil(GCH) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: argument types and order match the `colstats` kernel.
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.colstats)
                    .arg(&self.indptr)
                    .arg(&self.indices)
                    .arg(&self.vals)
                    .arg(&ni)
                    .arg(&pi)
                    .arg(&mut sum_d)
                    .arg(&mut sumsq_d)
                    .arg(&mut mn_d)
                    .arg(&mut mx_d)
                    .arg(&mut cnt_d)
                    .launch(cfg)
            }
            .map_err(|e| drv_err("launch colstats", e))?;
            gpu.download_into(&sum_d, &mut sum, "download sums")?;
            gpu.download_into(&sumsq_d, &mut sumsq, "download sums of squares")?;
            gpu.download_into(&mn_d, &mut mn, "download minima")?;
            gpu.download_into(&mx_d, &mut mx, "download maxima")?;
            gpu.download_into(&cnt_d, &mut cnt, "download counts")
        })?;
        Ok(StatsAcc {
            sum,
            sumsq,
            min: mn.iter().map(|&v| v as f64).collect(),
            max: mx.iter().map(|&v| v as f64).collect(),
            nnz: cnt.iter().map(|&c| c as u64).collect(),
        })
    }
}

impl CudaScaledOp {
    pub fn new(csr: CsrDev, mean: &[f64], inv_sd: &[f64], x_colsum: Vec<f64>, l: usize) -> Result<Self> {
        let CsrDev { gpu, n, p, indptr, indices, vals } = csr;
        if l > i32::MAX as usize {
            return Err(Error::gpu(format!("l = {l} exceeds the CUDA kernel index range")));
        }
        gpu.ensure_fits(8 * p + 4 * (3 * p * l + n * l + 2 * l), &format!("PCA on {n} cells x {p} genes (l = {l})"))?;
        guard("CUDA PCA setup", || {
            let mean_d = gpu.upload(&mean.iter().map(|&m| m as f32).collect::<Vec<f32>>(), "upload gene means")?;
            let inv_sd_d = gpu.upload(&inv_sd.iter().map(|&s| s as f32).collect::<Vec<f32>>(), "upload gene 1/sd")?;
            let vs = gpu.alloc_zeros::<f32>(p * l, "alloc vs")?;
            let shift = gpu.alloc_zeros::<f32>(l, "alloc shift")?;
            let colsum = gpu.alloc_zeros::<f32>(l, "alloc colsum")?;
            let dense_n = gpu.alloc_zeros::<f32>(n * l, "alloc cells x l")?;
            let acc = gpu.alloc_zeros::<f32>(p * l, "alloc accumulator")?;
            let out_p = gpu.alloc_zeros::<f32>(p * l, "alloc genes x l")?;
            Ok(Self {
                gpu: gpu.clone(),
                n,
                p,
                l,
                indptr,
                indices,
                vals,
                mean_d,
                inv_sd_d,
                mean: mean.to_vec(),
                inv_sd: inv_sd.to_vec(),
                x_colsum,
                vs: RefCell::new(vs),
                shift: RefCell::new(shift),
                colsum: RefCell::new(colsum),
                dense_n: RefCell::new(dense_n),
                acc: RefCell::new(acc),
                out_p: RefCell::new(out_p),
                host_n: RefCell::new(vec![0.0; n * l]),
                host_p: RefCell::new(vec![0.0; p * l]),
            })
        })
    }
}

/// Allocate `len` elements on the device and fill them through a bounded host staging
/// buffer, so the host never holds a full converted copy of the CSR.
fn upload_chunked<T: DeviceRepr>(gpu: &Gpu, len: usize, chunk: impl Fn(&std::ops::Range<usize>) -> Vec<T>, step: &str) -> Result<CudaSlice<T>> {
    // SAFETY: every element is written below before the slice is read.
    let mut dev = unsafe { gpu.stream.alloc::<T>(len) }.map_err(|e| drv_err(step, e))?;
    let mut start = 0;
    while start < len {
        let end = (start + STAGE).min(len);
        let host = chunk(&(start..end));
        let mut view = dev.slice_mut(start..end);
        gpu.stream.memcpy_htod(&host, &mut view).map_err(|e| drv_err(step, e))?;
        start = end;
    }
    gpu.sync(step)?;
    Ok(dev)
}

impl CudaScaledOp {
    /// Z^T Z (genes x genes, f64) in one pass over X: X^T X on the device (`xtx`), then the
    /// mean and 1/sd correction there too (`gram_finish`),
    /// `diag(inv_sd) (X^T X - mu s^T - s mu^T + n mu mu^T) diag(inv_sd)` with `s` the column
    /// sums of X (`s = n mu` unless the values were capped). Only the column-major lower
    /// triangle is written.
    fn gram_device(&self) -> Result<CudaSlice<f64>> {
        let (n, p) = (self.n, self.p);
        let gpu = &self.gpu;
        let mut g = gpu.alloc_zeros::<f64>(p * p, "alloc Gram matrix")?;
        let pi = p as i32;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: argument types and order match the `xtx` kernel; one block per row.
        unsafe { gpu.stream.launch_builder(&gpu.xtx).arg(&self.indptr).arg(&self.indices).arg(&self.vals).arg(&mut g).arg(&pi).launch(cfg) }
            .map_err(|e| drv_err("launch xtx", e))?;
        let mean = gpu.upload(&self.mean, "upload gene means")?;
        let colsum = gpu.upload(&self.x_colsum, "upload column sums")?;
        let inv_sd = gpu.upload(&self.inv_sd, "upload gene 1/sd")?;
        let nf = n as f64;
        // SAFETY: argument types and order match the `gram_finish` kernel; every vector has p entries.
        unsafe {
            gpu.stream.launch_builder(&gpu.gram_finish).arg(&mut g).arg(&mean).arg(&colsum).arg(&inv_sd).arg(&nf).arg(&pi).launch(Gpu::cfg_1d(p * p))
        }
        .map_err(|e| drv_err("launch gram_finish", e))?;
        gpu.sync("Gram matrix")?;
        Ok(g)
    }

    /// Z^T Z on the host, for the Krylov path above `EIG_MAX_GENES`.
    pub fn gram(&self) -> Result<Mat<f64>> {
        let p = self.p;
        self.gpu.ensure_fits(8 * p * p, &format!("Gram matrix of {p} genes"))?;
        let mut host = vec![0f64; p * p];
        guard("CUDA PCA Z^T Z", || {
            let g = self.gram_device()?;
            self.gpu.download_into(&g, &mut host, "download Z^T Z")
        })?;
        Ok(Mat::from_fn(p, p, |a, b| host[a.min(b) * p + a.max(b)]))
    }

    /// The top `b` eigenpairs of Z^T Z (eigenvalues descending), with the Gram matrix formed
    /// and decomposed (cuSOLVER `syevd`) on the device; only the `b` eigenvectors come back.
    pub fn gram_top_eigen(&self, b: usize) -> Result<(Mat<f64>, Vec<f64>)> {
        use cudarc::cusolver::{safe::DnHandle, sys};
        let p = self.p;
        assert!(b <= p);
        self.gpu.ensure_fits(16 * p * p, &format!("Gram eigendecomposition of {p} genes"))?;
        let mut vecs = vec![0f64; p * b];
        let mut vals = vec![0f64; p];
        guard("CUDA PCA Gram eigendecomposition", || {
            let gpu = &self.gpu;
            let solver_err = |step: &str, e: cudarc::cusolver::result::CusolverError| Error::gpu(format!("{step}: cuSOLVER {:?}", e.0));
            let mut g = self.gram_device()?;
            let pi = p as i32;
            let dn = DnHandle::new(gpu.stream.clone()).map_err(|e| solver_err("cusolverDnCreate", e))?;
            let mut w = gpu.alloc_zeros::<f64>(p, "alloc eigenvalues")?;
            let mut info = gpu.alloc_zeros::<i32>(1, "alloc info")?;
            let (mode, fill) = (sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER);
            let mut lwork = 0i32;
            {
                let (gp, _a) = g.device_ptr_mut(&gpu.stream);
                let (wp, _b) = w.device_ptr_mut(&gpu.stream);
                // SAFETY: g is p x p and w has p entries on this handle's stream.
                unsafe { sys::cusolverDnDsyevd_bufferSize(dn.cu(), mode, fill, pi, gp as *const f64, pi, wp as *const f64, &mut lwork) }
                    .result()
                    .map_err(|e| solver_err("syevd buffer size", e))?;
            }
            let mut work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize, "alloc syevd workspace")?;
            {
                let (gp, _a) = g.device_ptr_mut(&gpu.stream);
                let (wp, _b) = w.device_ptr_mut(&gpu.stream);
                let (kp, _c) = work.device_ptr_mut(&gpu.stream);
                let (ip, _d) = info.device_ptr_mut(&gpu.stream);
                // SAFETY: buffers as above, the workspace has lwork entries, info one.
                unsafe { sys::cusolverDnDsyevd(dn.cu(), mode, fill, pi, gp as *mut f64, pi, wp as *mut f64, kp as *mut f64, lwork, ip as *mut i32) }
                    .result()
                    .map_err(|e| solver_err("syevd", e))?;
            }
            let mut status = [0i32];
            gpu.download_into(&info, &mut status, "download syevd info")?;
            if status[0] != 0 {
                return Err(Error::gpu(format!("syevd did not converge (info {})", status[0])));
            }
            gpu.download_into(&w, &mut vals, "download eigenvalues")?;
            gpu.stream.memcpy_dtoh(&g.slice((p - b) * p..), &mut vecs).map_err(|e| drv_err("download eigenvectors", e))?;
            gpu.sync("download eigenvectors")
        })?;
        // syevd sorts ascending; column c of the result is eigenvector p - 1 - c.
        let v = Mat::from_fn(p, b, |j, c| vecs[(b - 1 - c) * p + j]);
        Ok((v, (0..b).map(|c| vals[p - 1 - c]).collect()))
    }
}

impl super::Operator for CudaScaledOp {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.p
    }

    fn apply(&self, v: MatRef<'_, f64>, mut out: MatMut<'_, f64>) -> Result<()> {
        let (n, p, l) = (self.n, self.p, self.l);
        assert!(v.nrows() == p && v.ncols() == l && out.nrows() == n && out.ncols() == l);
        // vs = diag(inv_sd) v (row-major) and shift = mu^T vs, both in f64 first.
        let mut vs64 = vec![0.0f64; p * l];
        for c in 0..l {
            let col = v.col(c);
            for j in 0..p {
                vs64[j * l + c] = col[j] * self.inv_sd[j];
            }
        }
        let mut shift64 = vec![0.0f64; l];
        for j in 0..p {
            let m = self.mean[j];
            if m != 0.0 {
                for (s, w) in shift64.iter_mut().zip(&vs64[j * l..(j + 1) * l]) {
                    *s += m * w;
                }
            }
        }
        let mut host_p = self.host_p.borrow_mut();
        for (h, &w) in host_p.iter_mut().zip(&vs64) {
            *h = w as f32;
        }
        let shift32: Vec<f32> = shift64.iter().map(|&s| s as f32).collect();
        let mut vs = self.vs.borrow_mut();
        let mut shift = self.shift.borrow_mut();
        let mut dense = self.dense_n.borrow_mut();
        let mut host_n = self.host_n.borrow_mut();
        guard("CUDA PCA Z v", || {
            self.gpu.upload_into(&host_p, &mut vs, "upload vs")?;
            self.gpu.upload_into(&shift32, &mut shift, "upload shift")?;
            let (ni, li) = (n as i32, l as i32);
            // SAFETY: argument types and order match the `zv` kernel signature; every buffer
            // has the length the kernel indexes.
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.zv)
                    .arg(&self.indptr)
                    .arg(&self.indices)
                    .arg(&self.vals)
                    .arg(&*vs)
                    .arg(&*shift)
                    .arg(&mut *dense)
                    .arg(&ni)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(n * l))
            }
            .map_err(|e| drv_err("launch zv", e))?;
            self.gpu.download_into(&dense, &mut host_n, "download Z v")
        })?;
        for c in 0..l {
            let src = &host_n[c * n..(c + 1) * n];
            let mut col = out.rb_mut().col_mut(c);
            for (r, &y) in src.iter().enumerate() {
                col[r] = y as f64;
            }
        }
        Ok(())
    }

    fn apply_t(&self, u: MatRef<'_, f64>) -> Result<Mat<f64>> {
        let (n, p, l) = (self.n, self.p, self.l);
        assert!(u.nrows() == n && u.ncols() == l);
        let mut host_n = self.host_n.borrow_mut();
        let mut colsum64 = vec![0.0f64; l];
        for c in 0..l {
            let col = u.col(c);
            let dst = &mut host_n[c * n..(c + 1) * n];
            let mut s = 0.0f64;
            for (r, y) in dst.iter_mut().enumerate() {
                let v = col[r];
                s += v;
                *y = v as f32;
            }
            colsum64[c] = s;
        }
        let colsum32: Vec<f32> = colsum64.iter().map(|&s| s as f32).collect();
        let mut dense = self.dense_n.borrow_mut();
        let mut colsum = self.colsum.borrow_mut();
        let mut acc = self.acc.borrow_mut();
        let mut out_p = self.out_p.borrow_mut();
        let mut host_p = self.host_p.borrow_mut();
        guard("CUDA PCA Z^T u", || {
            self.gpu.upload_into(&host_n, &mut dense, "upload u")?;
            self.gpu.upload_into(&colsum32, &mut colsum, "upload column sums")?;
            self.gpu.stream.memset_zeros(&mut *acc).map_err(|e| drv_err("zero accumulator", e))?;
            let (ni, pi, li) = (n as i32, p as i32, l as i32);
            // SAFETY: as in `apply`; the `ztu` and `ztu_finish` signatures are matched.
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.ztu)
                    .arg(&self.indptr)
                    .arg(&self.indices)
                    .arg(&self.vals)
                    .arg(&*dense)
                    .arg(&mut *acc)
                    .arg(&ni)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(n * l))
            }
            .map_err(|e| drv_err("launch ztu", e))?;
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.ztu_finish)
                    .arg(&*acc)
                    .arg(&self.mean_d)
                    .arg(&self.inv_sd_d)
                    .arg(&*colsum)
                    .arg(&mut *out_p)
                    .arg(&pi)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(p * l))
            }
            .map_err(|e| drv_err("launch ztu_finish", e))?;
            self.gpu.download_into(&out_p, &mut host_p, "download Z^T u")
        })?;
        Ok(Mat::from_fn(p, l, |j, c| host_p[j * l + c] as f64))
    }
}
