//! Leiden clustering: replaces Seurat's `FindClusters(algorithm = 4)`.
//!
//! CPU: single-clustering 0.7.0 (pinned), RB objective, which is what Seurat
//! (`RBConfigurationVertexPartition`) and Scanpy optimise (docs/RUST_ENGINE.md section 4).
//! Its Leiden loop is reproduced in [`leiden_rb`] with local moving and refinement replaced
//! by parallel, thread-count-independent versions (`cluster/local_move.rs`,
//! `cluster/refine.rs`) and its aggregation run in parallel (`cluster/aggregate.rs`). 500k synthetic cells (10 PCs,
//! k = 20, 12-core M-series): 5.0 s -> 1.8 s, ARI 0.999 against leidenalg on the same graph
//! at 100k.
//! GPU (Phase 7): cuGraph Leiden ([`crate::cugraph::CugraphLeiden`], feature `cugraph`) behind
//! the same `Clusterer` trait; [`leiden_on`] picks the backend from a [`LeidenDevice`].
//!
//! # Self-loops (the SNN diagonal)
//!
//! Seurat's `RNA_snn` keeps a diagonal of 1s, and how a self-loop enters the RB null model
//! depends on how the graph is handed to leidenalg:
//!
//! - Seurat 5.0.1 (verified: `RunLeiden` -> `leiden` R package -> python-igraph
//!   `Graph.Weighted_Adjacency`) converts the SNN matrix in igraph's default **directed**
//!   mode: every stored off-diagonal weight becomes an arc, so each edge is two arcs, and
//!   every diagonal 1 becomes **one** loop arc. (Seurat 5.1+ hands the matrix to `leidenbase`
//!   through `igraph::graph_from_adjacency_matrix`, also directed by default; not verified
//!   here, as only 5.0.1 is installed.) leidenalg's directed RB is
//!   `sum_c [w_c - g * K_c^out * K_c^in / m]`, so each loop adds 1 to a node's strength and
//!   1 to m.
//! - single-clustering is undirected with the igraph convention: a loop of weight w adds 2w
//!   to the node's strength and w to the total weight.
//!
//! Passing the diagonal at **half weight** makes the two objectives identical: with an
//! undirected loop of 0.5 the strength gains 1 and 4m = 2 * (2M + n), so
//! `Q_undirected = Q_directed / 2` up to a constant, for every partition. Keeping the
//! diagonal at full weight (strength + 2) or dropping it (strength + 0) optimises a
//! different objective from Seurat's. [`DIAGONAL_SCALE`] records the choice; the evidence
//! is in its documentation.

use std::cmp::Reverse;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::LeidenConfig;
use single_clustering::network::CSRNetwork;
use single_clustering::ClusteringError;

mod aggregate;
mod local_move;
mod refine;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::SparseGraph;

/// Parameter mapping from Seurat / the pipeline (docs/RUST_ENGINE.md section 4).
#[derive(Clone, Debug)]
pub struct ClusterParams {
    /// Seurat `resolution` (pipeline default 0.5).
    pub resolution: f64,
    /// Seurat `random.seed` (pipeline default 0).
    pub seed: u64,
    /// Seurat `n.iter` / leidenalg `n_iterations` (pipeline default 2).
    pub n_iterations: usize,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self { resolution: 0.5, seed: 0, n_iterations: 2 }
    }
}

/// A clustering backend. CPU (single-clustering) now; cuGraph in Phase 7.
pub trait Clusterer {
    fn name(&self) -> &'static str;
    /// One label per node, 0..n_clusters-1, numbered by decreasing cluster size (ties by
    /// lowest node index), like Seurat's `seurat_clusters` levels.
    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>>;
}

/// Factor applied to diagonal (self-loop) weights before they reach single-clustering.
///
/// 0.5 reproduces the objective Seurat's `FindClusters(algorithm = 4)` optimises on its SNN
/// (derivation in the module documentation). Graphs without a diagonal (scanpy
/// connectivities, or `SnnParams { keep_diagonal: false }`) are unaffected.
///
/// Evidence (17 Oct 2026, Seurat 5.0.1 + leidenalg 0.12 / python-igraph 1.0, 2-core VM):
///
/// - Same objective, numerically: for leidenalg's own partitions, RB quality on Seurat's
///   directed graph equals RB quality on the undirected half-loop graph to ~1e-12 relative
///   (PBMC 3k: 29982.8479199 vs 29982.8479199).
/// - Real PBMC 3k (2,638 cells, dims 1:10, k 20, Seurat's own SNN), 10 seeds per method,
///   mean cluster count:
///
///   | resolution | leidenalg as Seurat runs it | diag x 0.5 | diag x 1 | diag dropped |
///   |------------|-----------------------------|------------|----------|--------------|
///   | 0.5        | 8.7                         | 9.2        | 9.1      | 9.0          |
///   | 1.0        | 11.4                        | 11.3       | 12.0     | 11.1         |
///   | 2.0        | 17.7                        | 17.3       | 18.1     | 15.5         |
///
///   Dropping the diagonal under-splits and full weight over-splits as resolution grows;
///   half weight tracks Seurat at every resolution. Mean ARI against the leidenalg runs at
///   2.0: 0.789 (x 0.5), 0.791 (x 1), 0.744 (dropped), vs 0.762 between leidenalg's own seeds.
///   At the pipeline resolution 0.5 all three are within noise (ARI 0.889-0.896 vs leidenalg,
///   whose own seed-to-seed ARI is 0.851).
/// - Synthetic 2k / 5k bench data and the 500-cell fixture: every variant and seed gives
///   ARI 1.0 against Seurat's labels.
pub const DIAGONAL_SCALE: f32 = 0.5;

/// single-clustering Leiden with the RB objective.
pub struct SingleClusteringLeiden;

impl Clusterer for SingleClusteringLeiden {
    fn name(&self) -> &'static str {
        "single-clustering 0.7.0 (Leiden, RB)"
    }

    /// Errors: malformed CSR or a graph that isn't symmetric (`InvalidInput`), negative or
    /// non-finite resolution or `n_iterations == 0` (`InvalidParameter`).
    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
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
        if graph.n == 0 {
            return Ok(Vec::new());
        }

        // from_csr_parts takes owned buffers: this renumbered copy is the one copy per call
        let order = bfs_order(graph)?;
        let (indptr, indices, weights) = renumber(graph, &order);
        let network = CSRNetwork::from_csr_parts(indptr, indices, weights, None).map_err(map_err)?;

        // Leiden synchronises after every small batch of moves, so past a few threads per
        // ~60k nodes the wake-ups cost more than the work: 4 threads up to 250k nodes, 12 from
        // 750k (measured on AMD EPYC and Apple M: 16 threads were 1.8x slower than 4 at 100k).
        // Running inside the pool also keeps one-chunk parallel loops inline. Labels do not
        // depend on the thread count.
        let threads = (graph.n / LEIDEN_NODES_PER_THREAD).clamp(4, 12).min(rayon::current_num_threads());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error::Backend { context: "leiden thread pool", message: e.to_string() })?;
        let clustering = pool.install(|| leiden_rb(&network, params));
        let mut labels = vec![0usize; graph.n];
        for (&old, &c) in order.iter().zip(&clustering) {
            labels[old as usize] = c;
        }
        Ok(relabel_by_size(&labels))
    }
}

const LEIDEN_NODES_PER_THREAD: usize = 62_500;

/// single-clustering 0.7.0's `leiden` (RB, refinement on, its default tolerance, patience
/// and level cap) with its sequential local move and refinement replaced by the parallel
/// [`local_move::local_move`] and [`refine::refine`], and its aggregation by the parallel
/// (bit-identical) [`aggregate::aggregate`].
fn leiden_rb(graph: &CSRNetwork, params: &ClusterParams) -> Vec<usize> {
    let config = LeidenConfig::default();
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
    let n = graph.node_count();
    let mut labels: Vec<usize> = (0..n).collect();
    let mut stagnant = 0;
    for _ in 0..params.n_iterations {
        let mut level_rng = ChaCha8Rng::seed_from_u64(rng.next_u64());
        let mut coarse_graph: Option<CSRNetwork> = None;
        let mut membership = labels.clone();
        let mut node_at_level: Vec<usize> = (0..n).collect();
        let mut improvement = 0.0;
        let mut moved;
        let mut level = 0;
        loop {
            let level_graph = coarse_graph.as_ref().unwrap_or(graph);
            moved = local_move::local_move(level_graph, &membership, params.resolution, &mut level_rng);
            improvement += rb_quality(level_graph, &moved, params.resolution) - rb_quality(level_graph, &membership, params.resolution);
            level += 1;
            if level == config.max_levels {
                break;
            }
            let moved_u32: Vec<u32> = moved.iter().map(|&c| c as u32).collect();
            let refined = refine::refine(level_graph, &moved_u32, params.resolution, config.refine_randomness, &mut level_rng);
            let (group, n_groups) = renumber_ascending(&refined);
            let coarse = aggregate::aggregate(level_graph, &group, n_groups);
            if coarse.node_count() >= level_graph.node_count() {
                break;
            }
            membership = vec![0; coarse.node_count()];
            for (v, &g) in group.iter().enumerate() {
                membership[g as usize] = moved[v];
            }
            for slot in node_at_level.iter_mut() {
                *slot = group[*slot] as usize;
            }
            coarse_graph = Some(coarse);
        }
        let (final_ids, _) = renumber_ascending(&moved);
        labels = node_at_level.iter().map(|&v| final_ids[v] as usize).collect();
        if improvement <= config.tolerance * graph.total_weight().abs().max(1.0) {
            stagnant += 1;
            if stagnant >= config.patience {
                break;
            }
        } else {
            stagnant = 0;
        }
    }
    labels
}

/// single-clustering's RB quality `sum_c [internal_c - resolution * K_c^2 / 4m]` (internal
/// weight counting each edge and loop once), with the edge scan in parallel.
fn rb_quality(graph: &CSRNetwork, membership: &[usize], resolution: f64) -> f64 {
    let four_m = 4.0 * graph.total_weight();
    if four_m == 0.0 {
        return 0.0;
    }
    let internal: f64 = (0..graph.node_count())
        .into_par_iter()
        .map(|v| graph.neighbors(v).filter(|&(u, _)| u >= v && membership[u] == membership[v]).map(|(_, w)| w).sum::<f64>())
        .sum();
    let mut strength = vec![0f64; membership.iter().copied().max().map_or(0, |m| m + 1)];
    for (v, &c) in membership.iter().enumerate() {
        strength[c] += graph.strength(v);
    }
    internal - resolution * strength.iter().map(|k| k * k).sum::<f64>() / four_m
}

/// Community ids renumbered 0.. in ascending order of the old id (as single-clustering's
/// `Partition::renumber`), and the number of communities.
fn renumber_ascending(ids: &[usize]) -> (Vec<u32>, usize) {
    let mut new_id = vec![u32::MAX; ids.iter().copied().max().map_or(0, |m| m + 1)];
    for &c in ids {
        new_id[c] = 0;
    }
    let mut next = 0;
    for id in new_id.iter_mut().filter(|id| **id == 0) {
        *id = next;
        next += 1;
    }
    (ids.iter().map(|&c| new_id[c]).collect(), next as usize)
}

/// Nodes in breadth-first order (`order[new] = old`), restarting at the lowest unvisited
/// node. Graph neighbours end up close in memory, so the membership and community
/// lookups of each local move hit a few cache lines instead of ~2k scattered ones (the
/// input is in cell order, which is random with respect to the graph). 500k-cell SNN:
/// Errors on an index out of range.
fn bfs_order(g: &SparseGraph) -> Result<Vec<u32>> {
    let n = g.n;
    if let Some(&j) = g.indices.par_iter().find_first(|&&j| j as usize >= n) {
        return Err(Error::InvalidInput(format!("graph index {j} out of range for {n} nodes")));
    }
    // a bitset stays in L2 at 1M+ nodes, where a Vec<bool> does not
    let mut seen = vec![0u64; n.div_ceil(64)];
    let mut order = Vec::with_capacity(n);
    for root in 0..n {
        if seen[root / 64] & (1 << (root % 64)) != 0 {
            continue;
        }
        seen[root / 64] |= 1 << (root % 64);
        let mut head = order.len();
        order.push(root as u32);
        while head < order.len() {
            let v = order[head] as usize;
            head += 1;
            for &j in &g.indices[g.indptr[v]..g.indptr[v + 1]] {
                let (word, bit) = (j as usize / 64, 1u64 << (j % 64));
                if seen[word] & bit == 0 {
                    seen[word] |= bit;
                    order.push(j);
                }
            }
        }
    }
    Ok(order)
}

/// The graph with node `order[new]` renamed `new`: rows re-sorted by column, diagonal
/// weights scaled by [`DIAGONAL_SCALE`].
fn renumber(g: &SparseGraph, order: &[u32]) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    let mut inv = vec![0u32; g.n];
    for (new, &old) in order.iter().enumerate() {
        inv[old as usize] = new as u32;
    }
    let mut indptr = vec![0usize; g.n + 1];
    for (new, &old) in order.iter().enumerate() {
        indptr[new + 1] = indptr[new] + g.indptr[old as usize + 1] - g.indptr[old as usize];
    }
    let nnz = indptr[g.n];
    let mut rows: Vec<(u32, f32)> = vec![(0, 0.0); nnz];
    let mut parts = Vec::with_capacity(g.n);
    let mut rest = rows.as_mut_slice();
    for new in 0..g.n {
        let (head, tail) = rest.split_at_mut(indptr[new + 1] - indptr[new]);
        parts.push(head);
        rest = tail;
    }
    parts.into_par_iter().zip(order).enumerate().for_each(|(new, (row, &old))| {
        let (s, e) = (g.indptr[old as usize], g.indptr[old as usize + 1]);
        for (slot, (&j, &w)) in row.iter_mut().zip(g.indices[s..e].iter().zip(&g.weights[s..e])) {
            let jn = inv[j as usize];
            *slot = (jn, if jn as usize == new { w * DIAGONAL_SCALE } else { w });
        }
        row.sort_unstable_by_key(|&(j, _)| j);
    });
    let (indices, weights) = rows.into_iter().unzip();
    (indptr, indices, weights)
}

/// Convenience: CPU Leiden with the default backend. Unchanged by the `cugraph` feature;
/// use [`leiden_on`] to choose a device.
pub fn leiden(graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
    SingleClusteringLeiden.cluster(graph, params)
}

/// Where Leiden runs. The binding strings are `"cpu"`, `"auto"`, `"cuda"` (device 0) and
/// `"cuda:<index>"`; `"gpu"` is accepted as an alias of `"cuda"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LeidenDevice {
    /// single-clustering (the reference implementation).
    #[default]
    Cpu,
    /// cuGraph on device 0 when scfast-core was built with the `cugraph` feature **and**
    /// `libcugraph_c.so` loads **and** a CUDA device is usable; otherwise the CPU.
    Auto,
    /// cuGraph on the given CUDA device index; an error when that is not possible.
    Cuda(usize),
}

impl LeidenDevice {
    /// Parses the binding-level device string (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "cpu" => return Ok(Self::Cpu),
            "auto" => return Ok(Self::Auto),
            "cuda" | "gpu" => return Ok(Self::Cuda(0)),
            _ => {}
        }
        if let Some(idx) = t.strip_prefix("cuda:").or_else(|| t.strip_prefix("gpu:")) {
            if let Ok(i) = idx.parse::<usize>() {
                return Ok(Self::Cuda(i));
            }
        }
        Err(Error::InvalidParameter(format!("device must be \"auto\", \"cpu\", \"cuda\" or \"cuda:<index>\", got {s:?}")))
    }

    /// The string form the bindings report (`"cpu"`, `"auto"`, `"cuda:<index>"`).
    pub fn as_string(&self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Auto => "auto".into(),
            Self::Cuda(i) => format!("cuda:{i}"),
        }
    }
}

/// Whether this build can run Leiden on cuGraph at all (feature `cugraph` compiled in). It
/// says nothing about the library or a GPU being present: see [`resolve_leiden_device`].
pub const fn cugraph_compiled() -> bool {
    cfg!(feature = "cugraph")
}

/// Turns [`LeidenDevice::Auto`] into the device [`leiden_on`] would use (`Cpu`, or `Cuda(0)`
/// when cuGraph is built, loadable and a device is usable) and checks that an explicit
/// `Cuda(i)` is usable. `Cpu` and a usable `Cuda(i)` are returned unchanged.
pub fn resolve_leiden_device(device: LeidenDevice) -> Result<LeidenDevice> {
    match device {
        LeidenDevice::Cpu => Ok(LeidenDevice::Cpu),
        LeidenDevice::Auto => {
            #[cfg(feature = "cugraph")]
            {
                if crate::cugraph::CugraphLeiden::new(0).is_ok() {
                    return Ok(LeidenDevice::Cuda(0));
                }
            }
            Ok(LeidenDevice::Cpu)
        }
        LeidenDevice::Cuda(i) => {
            #[cfg(feature = "cugraph")]
            {
                crate::cugraph::CugraphLeiden::new(i).map(|_| LeidenDevice::Cuda(i))
            }
            #[cfg(not(feature = "cugraph"))]
            {
                Err(Error::Backend {
                    context: "cuGraph Leiden",
                    message: format!(
                        "device cuda:{i} requested but scfast-core was built without the `cugraph` feature \
                         (build with `--features cugraph`), or use device \"cpu\" / \"auto\""
                    ),
                })
            }
        }
    }
}

/// Leiden on the chosen device. `Cpu` is exactly [`leiden`]; `Auto` falls back to the CPU
/// whenever cuGraph is not built, not loadable or has no usable device; `Cuda(i)` is an error
/// in those cases. Labels follow the same numbering on every backend (by decreasing size).
pub fn leiden_on(graph: &SparseGraph, params: &ClusterParams, device: LeidenDevice) -> Result<Vec<u32>> {
    match resolve_leiden_device(device)? {
        LeidenDevice::Cuda(_i) => {
            #[cfg(feature = "cugraph")]
            {
                crate::cugraph::CugraphLeiden::new(_i)?.cluster(graph, params)
            }
            #[cfg(not(feature = "cugraph"))]
            {
                unreachable!("resolve_leiden_device never returns Cuda without the cugraph feature")
            }
        }
        _ => leiden(graph, params),
    }
}

/// Structural checks single-clustering would otherwise trip over (it derives n from
/// `indptr`, so a wrong `n` would go unnoticed). Index bounds, weight sign/finiteness and
/// symmetry are checked by `CSRNetwork::from_csr_parts`.
pub(crate) fn validate_shape(g: &SparseGraph) -> Result<()> {
    if g.indptr.len() != g.n + 1 {
        return Err(Error::InvalidInput(format!("indptr has {} entries, expected n + 1 = {}", g.indptr.len(), g.n + 1)));
    }
    if g.indices.len() != g.weights.len() {
        return Err(Error::InvalidInput(format!(
            "indices ({}) and weights ({}) differ in length",
            g.indices.len(),
            g.weights.len()
        )));
    }
    if g.indptr[0] != 0 || g.indptr[g.n] != g.indices.len() {
        return Err(Error::InvalidInput("indptr must start at 0 and end at nnz".into()));
    }
    if g.indptr.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::InvalidInput("indptr is not non-decreasing".into()));
    }
    Ok(())
}

fn map_err(e: ClusteringError) -> Error {
    match e {
        ClusteringError::InvalidConfig(m) => Error::InvalidParameter(m),
        ClusteringError::NodeIndexOutOfRange { .. }
        | ClusteringError::NonFiniteWeight { .. }
        | ClusteringError::NegativeWeight { .. }
        | ClusteringError::InvalidCsr(_)
        | ClusteringError::AsymmetricGraph { .. }
        | ClusteringError::TooManyNodes { .. } => Error::InvalidInput(format!("graph rejected by single-clustering: {e}")),
        other => Error::Backend { context: "single-clustering Leiden", message: other.to_string() },
    }
}

/// Renumber clusters 0.. by decreasing size, ties broken by the lowest node index in the
/// cluster, so labels don't depend on the backend's internal numbering.
pub(crate) fn relabel_by_size(labels: &[usize]) -> Vec<u32> {
    let n_slots = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut size = vec![0usize; n_slots];
    let mut first = vec![usize::MAX; n_slots];
    for (v, &c) in labels.iter().enumerate() {
        size[c] += 1;
        first[c] = first[c].min(v);
    }
    let mut order: Vec<usize> = (0..n_slots).filter(|&c| size[c] > 0).collect();
    order.sort_unstable_by_key(|&c| (Reverse(size[c]), first[c]));
    let mut new_id = vec![0u32; n_slots];
    for (rank, &c) in order.iter().enumerate() {
        new_id[c] = rank as u32;
    }
    labels.iter().map(|&c| new_id[c]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relabels_by_size_then_first_index() {
        // sizes: a=2 (first 0), b=3 (first 1), c=2 (first 3), d=1
        let raw = [7, 2, 2, 5, 7, 2, 5, 9];
        assert_eq!(relabel_by_size(&raw), vec![1, 0, 0, 2, 1, 0, 2, 3]);
        assert!(relabel_by_size(&[]).is_empty());
    }

    #[test]
    fn rejects_bad_shapes_and_params() {
        let g = SparseGraph { n: 2, indptr: vec![0, 1, 2], indices: vec![1, 0], weights: vec![1.0, 1.0] };
        assert!(leiden(&g, &ClusterParams::default()).is_ok());
        let bad_n = SparseGraph { n: 3, ..g.clone() };
        assert!(matches!(leiden(&bad_n, &ClusterParams::default()), Err(Error::InvalidInput(_))));
        let asym = SparseGraph { n: 2, indptr: vec![0, 1, 1], indices: vec![1], weights: vec![1.0] };
        assert!(matches!(leiden(&asym, &ClusterParams::default()), Err(Error::InvalidInput(_))));
        let p = ClusterParams { n_iterations: 0, ..Default::default() };
        assert!(matches!(leiden(&g, &p), Err(Error::InvalidParameter(_))));
        let p = ClusterParams { resolution: f64::NAN, ..Default::default() };
        assert!(matches!(leiden(&g, &p), Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn parses_device_strings() {
        assert_eq!(LeidenDevice::parse("CPU").unwrap(), LeidenDevice::Cpu);
        assert_eq!(LeidenDevice::parse(" auto ").unwrap(), LeidenDevice::Auto);
        assert_eq!(LeidenDevice::parse("cuda").unwrap(), LeidenDevice::Cuda(0));
        assert_eq!(LeidenDevice::parse("gpu").unwrap(), LeidenDevice::Cuda(0));
        assert_eq!(LeidenDevice::parse("cuda:3").unwrap(), LeidenDevice::Cuda(3));
        assert!(matches!(LeidenDevice::parse("tpu"), Err(Error::InvalidParameter(_))));
        assert!(matches!(LeidenDevice::parse("cuda:x"), Err(Error::InvalidParameter(_))));
        assert_eq!(LeidenDevice::Cuda(2).as_string(), "cuda:2");
    }

    /// Without a GPU (this machine), `auto` is the CPU and gives the CPU labels; an explicit
    /// CUDA device is a `Backend` error whether or not the feature is compiled in.
    #[test]
    fn device_dispatch_without_a_gpu() {
        let g = SparseGraph { n: 2, indptr: vec![0, 1, 2], indices: vec![1, 0], weights: vec![1.0, 1.0] };
        let p = ClusterParams::default();
        if std::env::var_os("SCFAST_CUGRAPH_TEST").is_some() {
            return; // a GPU box: the ignored tests in cugraph.rs cover it
        }
        assert_eq!(resolve_leiden_device(LeidenDevice::Cpu).unwrap(), LeidenDevice::Cpu);
        assert_eq!(resolve_leiden_device(LeidenDevice::Auto).unwrap(), LeidenDevice::Cpu);
        assert_eq!(leiden_on(&g, &p, LeidenDevice::Auto).unwrap(), leiden(&g, &p).unwrap());
        assert_eq!(leiden_on(&g, &p, LeidenDevice::Cpu).unwrap(), leiden(&g, &p).unwrap());
        assert!(matches!(leiden_on(&g, &p, LeidenDevice::Cuda(0)), Err(Error::Backend { .. })));
        assert!(matches!(resolve_leiden_device(LeidenDevice::Cuda(0)), Err(Error::Backend { .. })));
    }
}
