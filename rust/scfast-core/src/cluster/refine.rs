//! Leiden refinement (RB objective), parallel over the communities of the local-moving
//! partition, replacing single-clustering's sequential `refine` in [`super::leiden_rb`].
//!
//! Same rules as single-clustering 0.7.0: every node starts as its own refined community;
//! in a shuffled order, a node that is still a singleton may join a refined community of a
//! neighbour inside its own community, chosen among the moves that do not lose quality with
//! probability proportional to `exp(gain / randomness)`. A node only ever sees nodes of its
//! own community, so communities are refined independently: each one walks its nodes in the
//! order of one global shuffle and draws from its own ChaCha8 stream (`set_stream(c)`), so
//! the result does not depend on the thread count.

use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use single_clustering::network::CSRNetwork;

/// Refined membership of every node of `graph` (ids unique, below n, not consecutive),
/// refining `constraint` (community of each node).
pub(super) fn refine(graph: &CSRNetwork, constraint: &[u32], resolution: f64, randomness: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let n = graph.node_count();
    let four_m = 4.0 * graph.total_weight();
    let mean_edge_weight = if graph.edge_count() > 0 { graph.total_weight() / graph.edge_count() as f64 } else { 1.0 };
    let randomness = randomness * mean_edge_weight;

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let base_seed = rng.next_u64();

    // nodes grouped by community, each group in shuffled order (counting sort, stable)
    let slots = constraint.iter().max().map_or(0, |&c| c as usize + 1);
    let mut start = vec![0usize; slots + 1];
    for &c in constraint {
        start[c as usize + 1] += 1;
    }
    for c in 0..slots {
        start[c + 1] += start[c];
    }
    let mut members = vec![0u32; n];
    let mut fill = start.clone();
    for &v in &order {
        let c = constraint[v as usize] as usize;
        members[fill[c]] = v;
        fill[c] += 1;
    }
    // local[v] = position of v in its community's group
    let mut local = vec![0u32; n];
    for c in 0..slots {
        for (i, &v) in members[start[c]..start[c + 1]].iter().enumerate() {
            local[v as usize] = i as u32;
        }
    }

    let groups: Vec<usize> = (0..slots).filter(|&c| start[c + 1] > start[c]).collect();
    let refined_local: Vec<(usize, Vec<u32>)> = groups
        .par_iter()
        .map(|&c| {
            let nodes = &members[start[c]..start[c + 1]];
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
            rng.set_stream(c as u64);
            (c, refine_group(graph, constraint, &local, nodes, c as u32, resolution, randomness, four_m, &mut rng))
        })
        .collect();

    let mut refined = vec![0usize; n];
    for (c, ids) in refined_local {
        let nodes = &members[start[c]..start[c + 1]];
        for (&v, &r) in nodes.iter().zip(&ids) {
            refined[v as usize] = start[c] + r as usize;
        }
    }
    refined
}

/// Refines one community (`nodes`, in visiting order). Returns each node's refined id as a
/// position in `nodes`.
#[allow(clippy::too_many_arguments)]
fn refine_group(
    graph: &CSRNetwork,
    constraint: &[u32],
    local: &[u32],
    nodes: &[u32],
    own: u32,
    resolution: f64,
    randomness: f64,
    four_m: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<u32> {
    let m = nodes.len();
    let mut membership: Vec<u32> = (0..m as u32).collect();
    let mut strength: Vec<f64> = nodes.iter().map(|&v| graph.strength(v as usize)).collect();
    let mut size = vec![1u32; m];
    let mut weight_to = vec![0f64; m];
    let mut touched: Vec<u32> = Vec::new();
    let mut is_touched = vec![false; m];
    let mut candidates: Vec<(u32, f64)> = Vec::new();
    let mut exp_buf: Vec<f64> = Vec::new();
    if four_m == 0.0 {
        return membership;
    }

    for (i, &v) in nodes.iter().enumerate() {
        let me = membership[i];
        if size[me as usize] != 1 {
            continue;
        }
        let v = v as usize;
        for (u, w) in graph.neighbors(v) {
            if u == v || constraint[u] != own {
                continue;
            }
            let c = membership[local[u] as usize];
            if !is_touched[c as usize] {
                is_touched[c as usize] = true;
                touched.push(c);
            }
            weight_to[c as usize] += w;
        }
        // leaving a singleton empties it, so every gain is relative to a node on its own
        let k = graph.strength(v);
        candidates.clear();
        let mut best_rel = 0.0f64;
        for &c in &touched {
            if c == me {
                continue;
            }
            let rel = weight_to[c as usize] - resolution * k * 2.0 * strength[c as usize] / four_m;
            if rel >= 0.0 {
                candidates.push((c, rel));
                best_rel = best_rel.max(rel);
            }
        }
        for &c in &touched {
            weight_to[c as usize] = 0.0;
            is_touched[c as usize] = false;
        }
        touched.clear();
        if let Some(target) = select(&candidates, best_rel, randomness, rng, &mut exp_buf) {
            membership[i] = target;
            size[me as usize] = 0;
            size[target as usize] += 1;
            strength[target as usize] += strength[me as usize];
            strength[me as usize] = 0.0;
        }
    }
    membership
}

/// single-clustering's `select`: a candidate with probability proportional to
/// `exp((gain - best) / randomness)`; argmax (lowest id on ties) when `randomness <= 0` or
/// the weights degenerate.
fn select(candidates: &[(u32, f64)], best_rel: f64, randomness: f64, rng: &mut ChaCha8Rng, weights: &mut Vec<f64>) -> Option<u32> {
    if candidates.is_empty() {
        return None;
    }
    let argmax = || candidates.iter().filter(|&&(_, rel)| rel >= best_rel).map(|&(c, _)| c).min();
    if randomness <= 0.0 {
        return argmax();
    }
    weights.clear();
    weights.extend(candidates.iter().map(|&(_, rel)| ((rel - best_rel) / randomness).exp()));
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return argmax();
    }
    let mut threshold = rng.random::<f64>() * total;
    for (&(c, _), &w) in candidates.iter().zip(weights.iter()) {
        threshold -= w;
        if threshold <= 0.0 {
            return Some(c);
        }
    }
    candidates.last().map(|&(c, _)| c)
}
