//! GPU Leiden through cuGraph's C API (Phase 7, feature `cugraph`).
//!
//! `libcugraph_c.so` is loaded at **runtime** with `dlopen` (libloading), so this crate
//! builds, links and tests on a machine without CUDA, cuGraph or a GPU: nothing here is
//! resolved until a CUDA device is actually requested. The `extern "C"` signatures are written
//! by hand against the cuGraph **24.x** C API headers (`cugraph_c/resource_handle.h`,
//! `error.h`, `array.h`, `graph.h`, `random.h`, `community_algorithms.h`); they are not
//! generated from the headers, so every one of them is an assumption that only a run on a
//! RAPIDS box can confirm (the `#[ignore]` tests at the bottom are that run).
//!
//! # Mapping (docs/scfast-integration.md, Phase 7)
//!
//! | `ClusterParams` | `cugraph_leiden` argument |
//! |-----------------|---------------------------|
//! | `resolution`    | `resolution`              |
//! | `seed`          | `cugraph_rng_state_create(handle, seed)` |
//! | `n_iterations`  | — (`max_level = 100`, see [`MAX_LEVEL`]) |
//! | —               | `theta = 1.0`             |
//!
//! # Graph hand-off
//!
//! The [`SparseGraph`] is the full symmetric CSR (both directions of every edge). It is sent
//! as a COO edge list (`src`, `dst` as `int32`, `weights` as `float32`) with
//! `is_symmetric = TRUE`, which is what cuGraph's Python layer does for an undirected
//! `Graph`: the edge list already contains both arcs. The SNN diagonal is passed at **full
//! weight**: cuGraph stores a self-loop as a single arc, so a loop of weight w adds w to the
//! node's strength and w to the total arc weight, which is leidenalg's directed convention
//! (the one Seurat optimises, see `cluster.rs`); the half-weight scaling single-clustering
//! needs does not apply. The ignored ARI test checks this against the CPU backend.
//!
//! Nodes that cuGraph leaves out of the result (isolated vertices, if a graph without a
//! diagonal has any) each become their own cluster. Labels are renumbered by decreasing
//! cluster size exactly as on the CPU.
//!
//! # Library lookup
//!
//! `SCFAST_CUGRAPH_LIB=/path/to/libcugraph_c.so` wins; otherwise `libcugraph_c.so` is searched
//! the way `dlopen` does (`LD_LIBRARY_PATH`, the conda `lib/` directory when the process runs
//! inside the environment, the system paths). `cudaSetDevice` / `cudaGetDeviceCount` are
//! looked up through the same handle (cuGraph links the CUDA runtime) and then in
//! `libcudart.so.12` / `libcudart.so`; without them device 0 is used implicitly and a
//! non-zero device index is an error.
//!
//! # Graph construction entry point
//!
//! cuGraph 24.x has two SG graph constructors. `cugraph_sg_graph_create` (deprecated but
//! present through 24.x) has had the same 12-argument signature since 23.x, so it is used when
//! the symbol exists. Otherwise `cugraph_graph_create_sg` is used, whose signature changed
//! inside 24.x (24.10 inserted `symmetrize` before `do_expensive_check`). The layout is chosen
//! from `SCFAST_CUGRAPH_API` = `legacy` | `24.08` | `24.10` (default: `legacy` if the symbol
//! exists, else `24.10`). A wrong layout is undefined behaviour at the call, so set the
//! variable explicitly on a box whose cuGraph version is known.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

use libloading::Library;
use rayon::prelude::*;

use crate::cluster::{relabel_by_size, validate_shape, ClusterParams, Clusterer};
use crate::error::{Error, Result};
use crate::types::SparseGraph;

const CONTEXT: &str = "cuGraph Leiden";

// ------------------------------------------------------------------ C API types (assumed)

/// `cugraph_error_code_t` (enum, C int). `CUGRAPH_SUCCESS == 0`.
type ErrCode = c_int;
const CUGRAPH_SUCCESS: ErrCode = 0;
/// `bool_t` (enum `{ FALSE = 0, TRUE = 1 }`, C int).
type Bool = c_int;
const FALSE: Bool = 0;
const TRUE: Bool = 1;
/// `max_level` for `cugraph_leiden` (cuGraph's default). It caps the aggregation depth, which
/// is not leidenalg's `n_iterations` (full passes): mapping 2 iterations to 2 levels stops the
/// merge early and returns hundreds of fragments on a graph with 8 planted clusters.
const MAX_LEVEL: usize = 100;

/// `cugraph_data_type_id_t` (layout depends on the cuGraph version; see `dtypes_for_graph_create`).
type DType = c_int;

/// `cugraph_graph_properties_t { bool_t is_symmetric; bool_t is_multigraph; }`.
#[repr(C)]
struct GraphProperties {
    is_symmetric: Bool,
    is_multigraph: Bool,
}

/// Every cuGraph handle (`cugraph_resource_handle_t`, `cugraph_graph_t`, the arrays, views,
/// RNG state, results and errors) is an opaque `struct { int32_t align_; }`; they are only
/// ever passed back to cuGraph, so one opaque pointer type serves for all of them.
type Opaque = *mut c_void;

// resource_handle.h
type FnCreateResourceHandle = unsafe extern "C" fn(raft_handle: *mut c_void) -> Opaque;
type FnFreeResourceHandle = unsafe extern "C" fn(handle: Opaque);
// error.h
type FnErrorMessage = unsafe extern "C" fn(error: *const c_void) -> *const c_char;
type FnErrorFree = unsafe extern "C" fn(error: Opaque);
// array.h
type FnDeviceArrayCreate =
    unsafe extern "C" fn(handle: *const c_void, n_elems: usize, dtype: DType, array: *mut Opaque, error: *mut Opaque) -> ErrCode;
type FnDeviceArrayFree = unsafe extern "C" fn(array: Opaque);
type FnViewCreate = unsafe extern "C" fn(pointer: *mut c_void, n_elems: usize, dtype: DType) -> Opaque;
type FnDeviceArrayView = unsafe extern "C" fn(array: Opaque) -> Opaque;
type FnViewFree = unsafe extern "C" fn(view: Opaque);
type FnViewSize = unsafe extern "C" fn(view: *const c_void) -> usize;
type FnViewType = unsafe extern "C" fn(view: *const c_void) -> DType;
type FnViewCopyFromHost =
    unsafe extern "C" fn(handle: *const c_void, dst: Opaque, h_src: *const u8, error: *mut Opaque) -> ErrCode;
type FnViewCopyToHost =
    unsafe extern "C" fn(handle: *const c_void, h_dst: *mut u8, src: *const c_void, error: *mut Opaque) -> ErrCode;
// graph.h: cugraph_sg_graph_create (23.x-24.x, deprecated in 24.x)
type FnSgGraphCreateLegacy = unsafe extern "C" fn(
    handle: *const c_void,
    properties: *const GraphProperties,
    src: *const c_void,
    dst: *const c_void,
    weights: *const c_void,
    edge_ids: *const c_void,
    edge_type_ids: *const c_void,
    store_transposed: Bool,
    renumber: Bool,
    do_expensive_check: Bool,
    graph: *mut Opaque,
    error: *mut Opaque,
) -> ErrCode;
// graph.h: cugraph_graph_create_sg, 24.02-24.08 layout
type FnGraphCreateSg2408 = unsafe extern "C" fn(
    handle: *const c_void,
    properties: *const GraphProperties,
    vertices: *const c_void,
    src: *const c_void,
    dst: *const c_void,
    weights: *const c_void,
    edge_ids: *const c_void,
    edge_type_ids: *const c_void,
    store_transposed: Bool,
    renumber: Bool,
    drop_self_loops: Bool,
    drop_multi_edges: Bool,
    do_expensive_check: Bool,
    graph: *mut Opaque,
    error: *mut Opaque,
) -> ErrCode;
// graph.h: cugraph_graph_create_sg, 24.10+ layout (symmetrize added)
type FnGraphCreateSg2410 = unsafe extern "C" fn(
    handle: *const c_void,
    properties: *const GraphProperties,
    vertices: *const c_void,
    src: *const c_void,
    dst: *const c_void,
    weights: *const c_void,
    edge_ids: *const c_void,
    edge_type_ids: *const c_void,
    store_transposed: Bool,
    renumber: Bool,
    drop_self_loops: Bool,
    drop_multi_edges: Bool,
    symmetrize: Bool,
    do_expensive_check: Bool,
    graph: *mut Opaque,
    error: *mut Opaque,
) -> ErrCode;
type FnGraphFree = unsafe extern "C" fn(graph: Opaque);
// random.h
type FnRngStateCreate = unsafe extern "C" fn(handle: *const c_void, seed: u64, state: *mut Opaque, error: *mut Opaque) -> ErrCode;
type FnRngStateFree = unsafe extern "C" fn(state: Opaque);
// community_algorithms.h
type FnLeiden = unsafe extern "C" fn(
    handle: *const c_void,
    rng_state: Opaque,
    graph: Opaque,
    max_level: usize,
    resolution: f64,
    theta: f64,
    do_expensive_check: Bool,
    result: *mut Opaque,
    error: *mut Opaque,
) -> ErrCode;
type FnResultGetView = unsafe extern "C" fn(result: Opaque) -> Opaque;
type FnResultGetModularity = unsafe extern "C" fn(result: Opaque) -> f64;
type FnResultFree = unsafe extern "C" fn(result: Opaque);
// cuda_runtime_api.h
type FnCudaSetDevice = unsafe extern "C" fn(device: c_int) -> c_int;
type FnCudaGetDeviceCount = unsafe extern "C" fn(count: *mut c_int) -> c_int;

// ------------------------------------------------------------------ library loading

#[derive(Clone, Copy)]
enum GraphCreate {
    Legacy(FnSgGraphCreateLegacy),
    V2408(FnGraphCreateSg2408),
    V2410(FnGraphCreateSg2410),
}

/// The resolved function pointers. The `Library` is kept alive for the life of the process
/// (the `OnceLock` below), so the copied-out pointers stay valid.
fn dtypes_for_graph_create(create: &GraphCreate) -> (DType, DType, DType) {
    // RAPIDS 24.10+ / 26.x: INT8=0, INT16=1, INT32=2, INT64=3, …, FLOAT32=8 (cugraph_c/types.h).
    // Older cuGraph: INT32=0, INT64=1, FLOAT32=2.
    match create {
        GraphCreate::V2410(_) => (2, 3, 8),
        GraphCreate::Legacy(_) | GraphCreate::V2408(_) => (0, 1, 2),
    }
}

struct Api {
    _lib: Library,
    path: String,
    create_resource_handle: FnCreateResourceHandle,
    free_resource_handle: FnFreeResourceHandle,
    error_message: FnErrorMessage,
    error_free: FnErrorFree,
    device_array_create: FnDeviceArrayCreate,
    device_array_free: FnDeviceArrayFree,
    device_array_view: FnDeviceArrayView,
    view_create: FnViewCreate,
    view_free: FnViewFree,
    view_size: FnViewSize,
    view_type: FnViewType,
    view_copy_from_host: FnViewCopyFromHost,
    view_copy_to_host: FnViewCopyToHost,
    graph_create: GraphCreate,
    graph_create_name: &'static str,
    dtype_int32: DType,
    dtype_int64: DType,
    dtype_float32: DType,
    graph_free: FnGraphFree,
    rng_state_create: FnRngStateCreate,
    rng_state_free: FnRngStateFree,
    leiden: FnLeiden,
    result_get_vertices: FnResultGetView,
    result_get_clusters: FnResultGetView,
    result_get_modularity: FnResultGetModularity,
    result_free: FnResultFree,
    cuda_set_device: Option<FnCudaSetDevice>,
    cuda_get_device_count: Option<FnCudaGetDeviceCount>,
    /// Kept so its symbols stay valid when cudart was opened separately.
    _cudart: Option<Library>,
}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

fn backend(message: impl Into<String>) -> Error {
    Error::Backend { context: CONTEXT, message: message.into() }
}

/// Copies a symbol out of `lib` as a plain function pointer.
///
/// Safety: `T` must be the `unsafe extern "C" fn` type whose layout matches the symbol; that
/// is the module-level assumption documented above.
unsafe fn sym<T: Copy>(lib: &Library, name: &str) -> std::result::Result<T, String> {
    let cname = CString::new(name).map_err(|e| e.to_string())?;
    let s: libloading::Symbol<T> = lib.get(cname.as_bytes_with_nul()).map_err(|e| format!("{name}: {e}"))?;
    Ok(*s)
}

unsafe fn sym_opt<T: Copy>(lib: &Library, name: &str) -> Option<T> {
    sym::<T>(lib, name).ok()
}

fn library_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("SCFAST_CUGRAPH_LIB") {
        v.push(p.to_string_lossy().into_owned());
    }
    v.push("libcugraph_c.so".into());
    v
}

fn load_api() -> std::result::Result<Api, String> {
    let mut errors = Vec::new();
    let mut loaded = None;
    for cand in library_candidates() {
        // Safety: loading a shared library runs its initialisers; cuGraph's are the usual C++
        // static constructors.
        match unsafe { Library::new(&cand) } {
            Ok(lib) => {
                loaded = Some((lib, cand));
                break;
            }
            Err(e) => errors.push(format!("{cand}: {e}")),
        }
    }
    let Some((lib, path)) = loaded else {
        return Err(format!(
            "libcugraph_c.so not found ({}); install RAPIDS cuGraph >= 24.x or set SCFAST_CUGRAPH_LIB",
            errors.join("; ")
        ));
    };

    // Safety: each `T` is the hand-written signature for that symbol (see the module docs).
    unsafe {
        let legacy: Option<FnSgGraphCreateLegacy> = sym_opt(&lib, "cugraph_sg_graph_create");
        let modern = sym_opt::<FnGraphCreateSg2410>(&lib, "cugraph_graph_create_sg");
        let layout = std::env::var("SCFAST_CUGRAPH_API").unwrap_or_default().to_ascii_lowercase();
        let (graph_create, graph_create_name) = match layout.as_str() {
            "legacy" => (GraphCreate::Legacy(legacy.ok_or("SCFAST_CUGRAPH_API=legacy but cugraph_sg_graph_create is not exported")?), "cugraph_sg_graph_create"),
            "24.08" | "24.02" | "24.04" | "24.06" => {
                let f: FnGraphCreateSg2408 = sym(&lib, "cugraph_graph_create_sg")?;
                (GraphCreate::V2408(f), "cugraph_graph_create_sg (24.02-24.08 layout)")
            }
            "24.10" | "24.12" => (
                GraphCreate::V2410(modern.ok_or("SCFAST_CUGRAPH_API=24.10 but cugraph_graph_create_sg is not exported")?),
                "cugraph_graph_create_sg (24.10+ layout)",
            ),
            "" => match (legacy, modern) {
                (Some(f), _) => (GraphCreate::Legacy(f), "cugraph_sg_graph_create"),
                (None, Some(f)) => (GraphCreate::V2410(f), "cugraph_graph_create_sg (24.10+ layout, assumed)"),
                (None, None) => return Err("neither cugraph_sg_graph_create nor cugraph_graph_create_sg is exported".into()),
            },
            other => return Err(format!("SCFAST_CUGRAPH_API={other:?}: expected legacy, 24.08 or 24.10")),
        };
        let (dtype_int32, dtype_int64, dtype_float32) = dtypes_for_graph_create(&graph_create);
        let graph_free: FnGraphFree = sym_opt(&lib, "cugraph_graph_free")
            .or_else(|| sym_opt(&lib, "cugraph_sg_graph_free"))
            .ok_or("neither cugraph_graph_free nor cugraph_sg_graph_free is exported")?;

        let mut cudart: Option<Library> = None;
        let mut cuda_set_device: Option<FnCudaSetDevice> = sym_opt(&lib, "cudaSetDevice");
        let mut cuda_get_device_count: Option<FnCudaGetDeviceCount> = sym_opt(&lib, "cudaGetDeviceCount");
        if cuda_set_device.is_none() || cuda_get_device_count.is_none() {
            for name in ["libcudart.so.12", "libcudart.so"] {
                if let Ok(rt) = Library::new(name) {
                    cuda_set_device = cuda_set_device.or_else(|| sym_opt(&rt, "cudaSetDevice"));
                    cuda_get_device_count = cuda_get_device_count.or_else(|| sym_opt(&rt, "cudaGetDeviceCount"));
                    cudart = Some(rt);
                    break;
                }
            }
        }

        Ok(Api {
            path,
            create_resource_handle: sym(&lib, "cugraph_create_resource_handle")?,
            free_resource_handle: sym(&lib, "cugraph_free_resource_handle")?,
            error_message: sym(&lib, "cugraph_error_message")?,
            error_free: sym(&lib, "cugraph_error_free")?,
            device_array_create: sym(&lib, "cugraph_type_erased_device_array_create")?,
            device_array_free: sym(&lib, "cugraph_type_erased_device_array_free")?,
            device_array_view: sym(&lib, "cugraph_type_erased_device_array_view")?,
            view_create: sym(&lib, "cugraph_type_erased_device_array_view_create")?,
            view_free: sym(&lib, "cugraph_type_erased_device_array_view_free")?,
            view_size: sym(&lib, "cugraph_type_erased_device_array_view_size")?,
            view_type: sym(&lib, "cugraph_type_erased_device_array_view_type")?,
            view_copy_from_host: sym(&lib, "cugraph_type_erased_device_array_view_copy_from_host")?,
            view_copy_to_host: sym(&lib, "cugraph_type_erased_device_array_view_copy_to_host")?,
            graph_create,
            graph_create_name,
            dtype_int32,
            dtype_int64,
            dtype_float32,
            graph_free,
            rng_state_create: sym(&lib, "cugraph_rng_state_create")?,
            rng_state_free: sym(&lib, "cugraph_rng_state_free")?,
            leiden: sym(&lib, "cugraph_leiden")?,
            result_get_vertices: sym(&lib, "cugraph_hierarchical_clustering_result_get_vertices")?,
            result_get_clusters: sym(&lib, "cugraph_hierarchical_clustering_result_get_clusters")?,
            result_get_modularity: sym(&lib, "cugraph_hierarchical_clustering_result_get_modularity")?,
            result_free: sym(&lib, "cugraph_hierarchical_clustering_result_free")?,
            cuda_set_device,
            cuda_get_device_count,
            _cudart: cudart,
            _lib: lib,
        })
    }
}

fn api() -> Result<&'static Api> {
    match API.get_or_init(load_api) {
        Ok(a) => Ok(a),
        Err(e) => Err(backend(e.clone())),
    }
}

/// Path (or name) of the cuGraph library once it has been loaded, for diagnostics.
pub fn library_path() -> Option<&'static str> {
    API.get().and_then(|r| r.as_ref().ok()).map(|a| a.path.as_str())
}

// ------------------------------------------------------------------ RAII wrappers

struct Handle<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for Handle<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.free_resource_handle)(self.ptr) }
    }
}

struct DeviceArray<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for DeviceArray<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.device_array_free)(self.ptr) }
    }
}

struct View<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for View<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.view_free)(self.ptr) }
    }
}

struct Graph<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.graph_free)(self.ptr) }
    }
}

struct RngState<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for RngState<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.rng_state_free)(self.ptr) }
    }
}

struct ClusteringResult<'a> {
    api: &'a Api,
    ptr: Opaque,
}
impl Drop for ClusteringResult<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.result_free)(self.ptr) }
    }
}

impl Api {
    /// Turns a cuGraph return code (and the error object it filled in) into an `Error`,
    /// freeing the error object.
    fn check(&self, code: ErrCode, error: Opaque, what: &str) -> Result<()> {
        if code == CUGRAPH_SUCCESS {
            if !error.is_null() {
                unsafe { (self.error_free)(error) };
            }
            return Ok(());
        }
        let mut message = format!("{what} failed with cugraph_error_code {code}");
        if !error.is_null() {
            unsafe {
                let m = (self.error_message)(error);
                if !m.is_null() {
                    message.push_str(": ");
                    message.push_str(&CStr::from_ptr(m).to_string_lossy());
                }
                (self.error_free)(error);
            }
        }
        Err(backend(message))
    }

    fn set_device(&self, device: usize) -> Result<()> {
        let idx = c_int::try_from(device).map_err(|_| Error::InvalidParameter(format!("CUDA device index {device} is out of range")))?;
        if let Some(count_fn) = self.cuda_get_device_count {
            let mut count: c_int = 0;
            let rc = unsafe { count_fn(&mut count) };
            if rc != 0 {
                return Err(backend(format!("cudaGetDeviceCount failed with cudaError {rc}: no usable CUDA device")));
            }
            if idx >= count {
                return Err(backend(format!("CUDA device {device} requested but only {count} device(s) are visible")));
            }
        }
        match self.cuda_set_device {
            Some(f) => {
                let rc = unsafe { f(idx) };
                if rc != 0 {
                    return Err(backend(format!("cudaSetDevice({idx}) failed with cudaError {rc}")));
                }
                Ok(())
            }
            None if device == 0 => Ok(()), // CUDA's default device
            None => Err(backend(format!(
                "cudaSetDevice is not reachable (no libcudart.so), so only device 0 can be used, not {device}"
            ))),
        }
    }

    fn create_handle(&self) -> Result<Handle<'_>> {
        let ptr = unsafe { (self.create_resource_handle)(std::ptr::null_mut()) };
        if ptr.is_null() {
            return Err(backend("cugraph_create_resource_handle returned NULL (no usable CUDA device?)"));
        }
        Ok(Handle { api: self, ptr })
    }

    fn upload<T: Copy>(&self, h: &Handle<'_>, data: &[T], dtype: DType, what: &str) -> Result<(DeviceArray<'_>, View<'_>)> {
        let mut arr: Opaque = std::ptr::null_mut();
        let mut err: Opaque = std::ptr::null_mut();
        let rc = unsafe { (self.device_array_create)(h.ptr, data.len(), dtype, &mut arr, &mut err) };
        self.check(rc, err, &format!("cugraph_type_erased_device_array_create ({what})"))?;
        if arr.is_null() {
            return Err(backend(format!("device array for {what} is NULL")));
        }
        let arr = DeviceArray { api: self, ptr: arr };
        let view = unsafe { (self.device_array_view)(arr.ptr) };
        if view.is_null() {
            return Err(backend(format!("device array view for {what} is NULL")));
        }
        let view = View { api: self, ptr: view };
        if !data.is_empty() {
            let mut err: Opaque = std::ptr::null_mut();
            let rc = unsafe { (self.view_copy_from_host)(h.ptr, view.ptr, data.as_ptr() as *const u8, &mut err) };
            self.check(rc, err, &format!("copy_from_host ({what})"))?;
        }
        Ok((arr, view))
    }

    /// Downloads an INT32 or INT64 view as `i64`.
    fn download_ints(&self, h: &Handle<'_>, view: &View<'_>, what: &str) -> Result<Vec<i64>> {
        let n = unsafe { (self.view_size)(view.ptr) };
        let dtype = unsafe { (self.view_type)(view.ptr) };
        let mut err: Opaque = std::ptr::null_mut();
        match dtype {
            d if d == self.dtype_int32 => {
                let mut buf = vec![0i32; n];
                let rc = unsafe { (self.view_copy_to_host)(h.ptr, buf.as_mut_ptr() as *mut u8, view.ptr, &mut err) };
                self.check(rc, err, &format!("copy_to_host ({what})"))?;
                Ok(buf.into_iter().map(i64::from).collect())
            }
            d if d == self.dtype_int64 => {
                let mut buf = vec![0i64; n];
                let rc = unsafe { (self.view_copy_to_host)(h.ptr, buf.as_mut_ptr() as *mut u8, view.ptr, &mut err) };
                self.check(rc, err, &format!("copy_to_host ({what})"))?;
                Ok(buf)
            }
            other => Err(backend(format!("{what}: unexpected cugraph_data_type_id {other} (expected INT32 or INT64)"))),
        }
    }
}

// ------------------------------------------------------------------ the clusterer

/// Leiden on cuGraph (`cugraph_leiden`), behind [`Clusterer`].
///
/// `CugraphLeiden::new(device)` loads `libcugraph_c.so` (once per process), selects the CUDA
/// device and probes it by creating and freeing a resource handle, so a missing library or
/// GPU is reported from the constructor, as a [`Error::Backend`]. The default clusterer stays
/// the CPU one; `cluster::leiden_on` picks this one for `LeidenDevice::Cuda` / `Auto`.
#[derive(Clone, Copy, Debug)]
pub struct CugraphLeiden {
    device: usize,
}

impl CugraphLeiden {
    /// cuGraph Leiden on CUDA device `device` (0 is the first visible device).
    pub fn new(device: usize) -> Result<Self> {
        let api = api()?;
        api.set_device(device)?;
        let _probe = api.create_handle()?;
        Ok(Self { device })
    }

    /// The CUDA device index this instance runs on.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Last modularity reported by cuGraph is not kept (the trait returns labels only); this
    /// runs Leiden and returns `(labels, modularity)` for the parity tests.
    pub fn cluster_with_modularity(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<(Vec<u32>, f64)> {
        if !params.resolution.is_finite() || params.resolution < 0.0 {
            return Err(Error::InvalidParameter(format!(
                "resolution must be finite and non-negative, got {}",
                params.resolution
            )));
        }
        if params.n_iterations == 0 {
            return Err(Error::InvalidParameter("n_iterations must be at least 1".into()));
        }
        validate_shape(graph)?;
        validate_edges(graph)?;
        if graph.n == 0 {
            return Ok((Vec::new(), 0.0));
        }
        if graph.n > i32::MAX as usize {
            return Err(Error::InvalidInput(format!("{} nodes exceed cuGraph's int32 vertex range", graph.n)));
        }
        if graph.nnz() == 0 {
            // No edges: every node is its own cluster (cuGraph would reject an empty edge list).
            return Ok(((0..graph.n as u32).collect(), 0.0));
        }

        // COO edge list from the full symmetric CSR: both arcs of every edge, loops once.
        let nnz = graph.nnz();
        let mut src = Vec::with_capacity(nnz);
        for i in 0..graph.n {
            src.extend(std::iter::repeat_n(i as i32, graph.indptr[i + 1] - graph.indptr[i]));
        }
        let dst: Vec<i32> = graph.indices.iter().map(|&j| j as i32).collect();
        let weights: &[f32] = &graph.weights;

        let api = api()?;
        api.set_device(self.device)?;
        let h = api.create_handle()?;
        let (_src_arr, src_view) = api.upload(&h, &src, api.dtype_int32, "src")?;
        let (_dst_arr, dst_view) = api.upload(&h, &dst, api.dtype_int32, "dst")?;
        let (_w_arr, w_view) = api.upload(&h, weights, api.dtype_float32, "weights")?;
        run(api, &h, graph.n, &src_view, &dst_view, &w_view, params)
    }

    /// [`cluster_with_modularity`](Self::cluster_with_modularity) on a graph that is already
    /// on the device, as COO arrays of `nnz` int32 sources, int32 destinations and f32
    /// weights holding both directions of every edge. No copy is made and nothing is
    /// validated beyond the sizes: the caller guarantees indices below `n`, a symmetric edge
    /// list and finite, non-negative weights, and that all work writing the arrays has
    /// finished.
    ///
    /// # Safety
    /// `src`, `dst` and `weights` must be device pointers to at least `nnz` elements of the
    /// stated types on this instance's device, valid for the duration of the call.
    pub unsafe fn cluster_device_coo(
        &self,
        n: usize,
        nnz: usize,
        src: *mut c_void,
        dst: *mut c_void,
        weights: *mut c_void,
        params: &ClusterParams,
    ) -> Result<(Vec<u32>, f64)> {
        if !params.resolution.is_finite() || params.resolution < 0.0 {
            return Err(Error::InvalidParameter(format!("resolution must be finite and non-negative, got {}", params.resolution)));
        }
        if n > i32::MAX as usize {
            return Err(Error::InvalidInput(format!("{n} nodes exceed cuGraph's int32 vertex range")));
        }
        if n == 0 {
            return Ok((Vec::new(), 0.0));
        }
        if nnz == 0 {
            return Ok(((0..n as u32).collect(), 0.0));
        }
        let api = api()?;
        api.set_device(self.device)?;
        let h = api.create_handle()?;
        let view = |ptr: *mut c_void, dtype: DType, what: &str| -> Result<View<'_>> {
            // SAFETY: the caller guarantees `ptr` holds `nnz` elements of `dtype`.
            let v = unsafe { (api.view_create)(ptr, nnz, dtype) };
            if v.is_null() {
                return Err(backend(format!("device array view for {what} is NULL")));
            }
            Ok(View { api, ptr: v })
        };
        let (src_view, dst_view, w_view) = (view(src, api.dtype_int32, "src")?, view(dst, api.dtype_int32, "dst")?, view(weights, api.dtype_float32, "weights")?);
        run(api, &h, n, &src_view, &dst_view, &w_view, params)
    }
}

/// Builds the cuGraph graph from the edge views and runs Leiden; returns the labels (numbered
/// as the CPU backend numbers them) and cuGraph's modularity.
fn run(api: &Api, h: &Handle<'_>, n: usize, src_view: &View<'_>, dst_view: &View<'_>, w_view: &View<'_>, params: &ClusterParams) -> Result<(Vec<u32>, f64)> {
    // RAPIDS 26+: NULL vertices defaults the vertex dtype to INT8; pass 0..n-1 as INT32 explicitly.
    let verts: Vec<i32> = (0..n as i32).collect();
    let (_vert_arr, vert_view) = api.upload(h, &verts, api.dtype_int32, "vertices")?;
    let props = GraphProperties { is_symmetric: TRUE, is_multigraph: FALSE };
    let mut g: Opaque = std::ptr::null_mut();
    let mut err: Opaque = std::ptr::null_mut();
    let null = std::ptr::null();
    let rc = unsafe {
        match api.graph_create {
            GraphCreate::Legacy(f) => f(
                h.ptr, &props, src_view.ptr, dst_view.ptr, w_view.ptr, null, null,
                FALSE, // store_transposed
                TRUE,  // renumber
                FALSE, // do_expensive_check
                &mut g, &mut err,
            ),
            GraphCreate::V2408(f) => f(
                h.ptr, &props, vert_view.ptr, src_view.ptr, dst_view.ptr, w_view.ptr, null, null,
                FALSE, // store_transposed
                TRUE,  // renumber
                FALSE, // drop_self_loops
                FALSE, // drop_multi_edges
                FALSE, // do_expensive_check
                &mut g, &mut err,
            ),
            GraphCreate::V2410(f) => f(
                h.ptr, &props, vert_view.ptr, src_view.ptr, dst_view.ptr, w_view.ptr, null, null,
                FALSE, // store_transposed
                TRUE,  // renumber
                FALSE, // drop_self_loops
                FALSE, // drop_multi_edges
                FALSE, // symmetrize (both arcs are already present)
                FALSE, // do_expensive_check
                &mut g, &mut err,
            ),
        }
    };
    api.check(rc, err, api.graph_create_name)?;
    if g.is_null() {
        return Err(backend("graph handle is NULL"));
    }
    let g = Graph { api, ptr: g };

    let mut rng: Opaque = std::ptr::null_mut();
    let mut err: Opaque = std::ptr::null_mut();
    let rc = unsafe { (api.rng_state_create)(h.ptr, params.seed, &mut rng, &mut err) };
    api.check(rc, err, "cugraph_rng_state_create")?;
    let rng = RngState { api, ptr: rng };

    let mut result: Opaque = std::ptr::null_mut();
    let mut err: Opaque = std::ptr::null_mut();
    let rc = unsafe {
        (api.leiden)(h.ptr, rng.ptr, g.ptr, MAX_LEVEL, params.resolution, 1.0, FALSE, &mut result, &mut err)
    };
    api.check(rc, err, "cugraph_leiden")?;
    if result.is_null() {
        return Err(backend("cugraph_leiden returned a NULL result"));
    }
    let result = ClusteringResult { api, ptr: result };

    let vertices = View { api, ptr: unsafe { (api.result_get_vertices)(result.ptr) } };
    let clusters = View { api, ptr: unsafe { (api.result_get_clusters)(result.ptr) } };
    if vertices.ptr.is_null() || clusters.ptr.is_null() {
        return Err(backend("result vertex/cluster view is NULL"));
    }
    let modularity = unsafe { (api.result_get_modularity)(result.ptr) };
    let vert = api.download_ints(&h, &vertices, "result vertices")?;
    let clus = api.download_ints(&h, &clusters, "result clusters")?;
    if vert.len() != clus.len() {
        return Err(backend(format!("result has {} vertices but {} cluster ids", vert.len(), clus.len())));
    }

    let labels = assemble_labels(n, &vert, &clus)?;
    Ok((labels, modularity))
}

impl Clusterer for CugraphLeiden {
    fn name(&self) -> &'static str {
        "cuGraph Leiden (C API, RB/modularity)"
    }

    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
        self.cluster_with_modularity(graph, params).map(|(l, _)| l)
    }
}

/// Index bounds, weight sign/finiteness and symmetry: the checks single-clustering does on
/// the CPU path, so both backends reject the same graphs with `InvalidInput`.
fn validate_edges(g: &SparseGraph) -> Result<()> {
    if let Some(&j) = g.indices.iter().find(|&&j| j as usize >= g.n) {
        return Err(Error::InvalidInput(format!("node index {j} out of range for {} nodes", g.n)));
    }
    if let Some(w) = g.weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
        return Err(Error::InvalidInput(format!("graph weights must be finite and non-negative, found {w}")));
    }
    let row = |j: usize| {
        let (js, je) = (g.indptr[j], g.indptr[j + 1]);
        (&g.indices[js..je], &g.weights[js..je])
    };
    let asymmetric = (0..g.n).into_par_iter().find_map_first(|i| {
        let (cols, ws) = row(i);
        cols.iter().zip(ws).find_map(|(&j, &w)| {
            if j as usize == i {
                return None;
            }
            let (back_cols, back_ws) = row(j as usize);
            // Canonical rows are sorted: binary search; otherwise scan.
            let pos = if back_cols.is_sorted() {
                back_cols.binary_search(&(i as u32)).ok()
            } else {
                back_cols.iter().position(|&k| k as usize == i)
            };
            let back = pos.map(|p| back_ws[p]);
            match back {
                Some(wb) if (wb - w).abs() <= 1e-6 * w.abs().max(wb.abs()).max(1.0) => None,
                _ => Some((i, j, w, back)),
            }
        })
    });
    if let Some((i, j, w, back)) = asymmetric {
        return Err(Error::InvalidInput(format!(
            "graph is not symmetric: edge {i}->{j} has weight {w} but {j}->{i} is {}",
            back.map_or("missing".to_string(), |b| b.to_string())
        )));
    }
    Ok(())
}

/// Turns cuGraph's (vertex, cluster) pairs into one label per node, each node cuGraph left
/// out getting a cluster of its own, renumbered by decreasing size like the CPU backend.
fn assemble_labels(n: usize, vertices: &[i64], clusters: &[i64]) -> Result<Vec<u32>> {
    const UNSET: usize = usize::MAX;
    let mut raw = vec![UNSET; n];
    let mut max_c: i64 = -1;
    for (&v, &c) in vertices.iter().zip(clusters) {
        if v < 0 || v as usize >= n {
            return Err(backend(format!("result vertex id {v} outside 0..{n}")));
        }
        if c < 0 {
            return Err(backend(format!("negative cluster id {c} for vertex {v}")));
        }
        if raw[v as usize] != UNSET {
            return Err(backend(format!("vertex {v} appears twice in the result")));
        }
        raw[v as usize] = c as usize;
        max_c = max_c.max(c);
    }
    let mut next = (max_c + 1) as usize;
    for r in raw.iter_mut() {
        if *r == UNSET {
            *r = next;
            next += 1;
        }
    }
    Ok(relabel_by_size(&raw))
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{leiden, LeidenDevice};
    use std::collections::HashMap;

    #[test]
    fn assembles_labels_with_missing_vertices() {
        // vertices 0,1,3 clustered (0,1 -> 5; 3 -> 2); vertex 2 missing -> its own cluster
        let l = assemble_labels(4, &[3, 0, 1], &[2, 5, 5]).unwrap();
        assert_eq!(l, vec![0, 0, 1, 2]); // size 2 first, then ties by first index (2 before 3)
        assert!(assemble_labels(2, &[2], &[0]).is_err());
        assert!(assemble_labels(2, &[0, 0], &[0, 0]).is_err());
        assert!(assemble_labels(2, &[0], &[-1]).is_err());
    }

    #[test]
    fn rejects_asymmetric_and_bad_edges() {
        let ok = SparseGraph { n: 2, indptr: vec![0, 1, 2], indices: vec![1, 0], weights: vec![1.0, 1.0] };
        assert!(validate_edges(&ok).is_ok());
        let asym = SparseGraph { n: 2, indptr: vec![0, 1, 1], indices: vec![1], weights: vec![1.0] };
        assert!(matches!(validate_edges(&asym), Err(Error::InvalidInput(_))));
        let neg = SparseGraph { weights: vec![-1.0, -1.0], ..ok.clone() };
        assert!(matches!(validate_edges(&neg), Err(Error::InvalidInput(_))));
        let oob = SparseGraph { indices: vec![1, 5], ..ok.clone() };
        assert!(matches!(validate_edges(&oob), Err(Error::InvalidInput(_))));
    }

    /// On a machine without cuGraph the constructor must fail cleanly (no panic, no abort)
    /// and name the library; with cuGraph present it must succeed. Either way nothing crashes.
    #[test]
    fn constructor_fails_cleanly_without_cugraph() {
        match CugraphLeiden::new(0) {
            Ok(c) => assert_eq!(c.device(), 0),
            Err(Error::Backend { context, message }) => {
                assert_eq!(context, CONTEXT);
                assert!(message.contains("libcugraph_c.so") || message.contains("CUDA") || message.contains("cugraph"), "{message}");
            }
            Err(other) => panic!("unexpected error kind: {other}"),
        }
    }

    // ---- GPU-only tests (L4 box with RAPIDS cuGraph 24.x) ----
    //
    // Run with:
    //   cd rust && SCFAST_CUGRAPH_TEST=1 cargo test --features cugraph -- --ignored cugraph
    // Set SCFAST_CUGRAPH_LIB=/path/to/libcugraph_c.so when it is not on the loader path
    // (conda: $CONDA_PREFIX/lib/libcugraph_c.so) and SCFAST_CUGRAPH_API=legacy|24.08|24.10
    // to force the graph-constructor layout for the installed cuGraph version.

    fn ari<A: Copy + Eq + std::hash::Hash, B: Copy + Eq + std::hash::Hash>(a: &[A], b: &[B]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut joint: HashMap<(A, B), f64> = HashMap::new();
        let mut ma: HashMap<A, f64> = HashMap::new();
        let mut mb: HashMap<B, f64> = HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            *joint.entry((x, y)).or_default() += 1.0;
            *ma.entry(x).or_default() += 1.0;
            *mb.entry(y).or_default() += 1.0;
        }
        let c2 = |v: f64| v * (v - 1.0) / 2.0;
        let index: f64 = joint.values().map(|&v| c2(v)).sum();
        let (sa, sb): (f64, f64) = (ma.values().map(|&v| c2(v)).sum(), mb.values().map(|&v| c2(v)).sum());
        let expected = sa * sb / c2(a.len() as f64);
        let max = (sa + sb) / 2.0;
        if max == expected {
            1.0
        } else {
            (index - expected) / (max - expected)
        }
    }

    /// kNN-style planted-partition graph: `n` nodes in `k` communities of unequal size, each
    /// node linked to ~15 nodes of its own community and ~2 outside, symmetric, unit diagonal
    /// (like an SNN graph). Deterministic.
    fn planted_graph(n: usize, k: usize, seed: u64) -> (SparseGraph, Vec<u32>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let truth: Vec<u32> = (0..n).map(|i| (((i * i) % 97 + i) % k) as u32).collect();
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &c) in truth.iter().enumerate() {
            members[c as usize].push(i);
        }
        let mut adj: Vec<HashMap<usize, f32>> = vec![HashMap::new(); n];
        for i in 0..n {
            adj[i].insert(i, 1.0);
            let own = &members[truth[i] as usize];
            for _ in 0..15 {
                let j = own[rng.random_range(0..own.len())];
                if j != i {
                    let w = 0.3 + 0.7 * rng.random::<f32>();
                    adj[i].insert(j, w);
                    adj[j].insert(i, w);
                }
            }
            for _ in 0..2 {
                let j = rng.random_range(0..n);
                if j != i {
                    let w = 0.05 + 0.1 * rng.random::<f32>();
                    adj[i].insert(j, w);
                    adj[j].insert(i, w);
                }
            }
        }
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n {
            let mut row: Vec<(usize, f32)> = adj[i].iter().map(|(&j, &w)| (j, w)).collect();
            row.sort_by_key(|&(j, _)| j);
            for (j, w) in row {
                indices.push(j as u32);
                weights.push(w);
            }
            indptr[i + 1] = indices.len();
        }
        (SparseGraph { n, indptr, indices, weights }, truth)
    }

    /// Phase 7 done-when: ARI(cuGraph, CPU Leiden) >= 0.9 on the same graph, and both close
    /// to the planted truth.
    #[test]
    #[ignore = "needs a CUDA GPU and RAPIDS cuGraph 24.x: SCFAST_CUGRAPH_TEST=1 cargo test --features cugraph -- --ignored cugraph"]
    fn cugraph_matches_cpu_leiden() {
        let gpu = CugraphLeiden::new(0).expect("cuGraph not loadable: set SCFAST_CUGRAPH_LIB");
        let (g, truth) = planted_graph(5000, 8, 42);
        let params = ClusterParams { resolution: 0.5, seed: 0, n_iterations: 2 };
        let cpu = leiden(&g, &params).unwrap();
        let (gpu_labels, modularity) = gpu.cluster_with_modularity(&g, &params).unwrap();
        assert_eq!(gpu_labels.len(), g.n);
        let a_cpu_gpu = ari(&cpu, &gpu_labels);
        let a_gpu_truth = ari(&gpu_labels, &truth);
        eprintln!(
            "cuGraph: {} clusters, modularity {modularity:.4}; ARI vs CPU {a_cpu_gpu:.3}, vs truth {a_gpu_truth:.3} ({})",
            gpu_labels.iter().max().map_or(0, |m| m + 1),
            library_path().unwrap_or("?")
        );
        assert!(a_cpu_gpu >= 0.9, "ARI cuGraph vs CPU Leiden {a_cpu_gpu} < 0.9");
        assert!(a_gpu_truth >= 0.9, "ARI cuGraph vs planted truth {a_gpu_truth} < 0.9");
    }

    /// Same seed, same graph -> identical labels; `auto` resolves to cuda:0 on a GPU box.
    #[test]
    #[ignore = "needs a CUDA GPU and RAPIDS cuGraph 24.x (see cugraph_matches_cpu_leiden)"]
    fn cugraph_is_deterministic_and_auto_picks_it() {
        let gpu = CugraphLeiden::new(0).unwrap();
        let (g, _) = planted_graph(2000, 5, 7);
        let params = ClusterParams::default();
        let a = gpu.cluster(&g, &params).unwrap();
        let b = gpu.cluster(&g, &params).unwrap();
        assert_eq!(a, b);
        assert_eq!(crate::cluster::resolve_leiden_device(LeidenDevice::Auto).unwrap(), LeidenDevice::Cuda(0));
        assert_eq!(crate::cluster::leiden_on(&g, &params, LeidenDevice::Cuda(0)).unwrap(), a);
    }
}
