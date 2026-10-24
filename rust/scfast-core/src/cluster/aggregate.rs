//! single-clustering's `CSRNetwork::aggregate` with the per-group accumulation run in
//! parallel. Each group still sums its members' edges in the same order and the pairs are
//! scattered in the same order, so the coarse graph is identical to single-clustering's.

use std::cell::RefCell;

use rayon::prelude::*;
use single_clustering::network::CSRNetwork;

/// Groups per parallel task.
const CHUNK: usize = 256;

thread_local! {
    static ACC: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Collapses each group (`group[v]`, below `n_groups`, every group non-empty) into one node:
/// node weights summed, a group's internal weight becomes a self-loop, weight between groups
/// an edge.
pub(super) fn aggregate(graph: &CSRNetwork, group: &[u32], n_groups: usize) -> CSRNetwork {
    let n = graph.node_count();

    let mut node_weights = vec![0.0f64; n_groups];
    for (v, &g) in group.iter().enumerate() {
        node_weights[g as usize] += graph.node_weight(v);
    }
    let mut starts = vec![0usize; n_groups + 1];
    for &g in group {
        starts[g as usize + 1] += 1;
    }
    for g in 0..n_groups {
        starts[g + 1] += starts[g];
    }
    let mut members = vec![0u32; n];
    let mut cursor = starts.clone();
    for (v, &g) in group.iter().enumerate() {
        members[cursor[g as usize]] = v as u32;
        cursor[g as usize] += 1;
    }

    // per group, accumulate to groups with id >= its own so each pair is seen once
    let pairs: Vec<Vec<(u32, u32, f32)>> = (0..n_groups.div_ceil(CHUNK))
        .into_par_iter()
        .map(|chunk| {
            ACC.with_borrow_mut(|acc| {
                if acc.len() < n_groups {
                    acc.resize(n_groups, 0.0);
                }
                let mut touched: Vec<usize> = Vec::new();
                let mut pairs = Vec::new();
                for g in chunk * CHUNK..((chunk + 1) * CHUNK).min(n_groups) {
                    let mut self_loop_weight = 0.0f64;
                    for &v in &members[starts[g]..starts[g + 1]] {
                        for (u, w) in graph.neighbors(v as usize) {
                            if u == v as usize {
                                self_loop_weight += w;
                            }
                            let h = group[u] as usize;
                            if h < g {
                                continue;
                            }
                            if acc[h] == 0.0 {
                                touched.push(h);
                            }
                            acc[h] += w;
                        }
                    }
                    for &h in &touched {
                        let w = acc[h];
                        acc[h] = 0.0;
                        if w <= 0.0 {
                            continue;
                        }
                        if h == g {
                            // regular edges seen from both ends, loops once: acc = 2 regular + loops
                            pairs.push((g as u32, g as u32, (0.5 * (w + self_loop_weight)) as f32));
                        } else {
                            pairs.push((g as u32, h as u32, w as f32));
                        }
                    }
                    touched.clear();
                }
                pairs
            })
        })
        .collect();

    let mut counts = vec![0usize; n_groups];
    for &(a, b, _) in pairs.iter().flatten() {
        counts[a as usize] += 1;
        if a != b {
            counts[b as usize] += 1;
        }
    }
    let mut node_ptrs = vec![0usize; n_groups + 1];
    for g in 0..n_groups {
        node_ptrs[g + 1] = node_ptrs[g] + counts[g];
    }
    let slots = node_ptrs[n_groups];
    let mut neighbors = vec![0u32; slots];
    let mut weights = vec![0.0f32; slots];
    let mut cursor = node_ptrs.clone();
    for &(a, b, w) in pairs.iter().flatten() {
        neighbors[cursor[a as usize]] = b;
        weights[cursor[a as usize]] = w;
        cursor[a as usize] += 1;
        if a != b {
            neighbors[cursor[b as usize]] = a;
            weights[cursor[b as usize]] = w;
            cursor[b as usize] += 1;
        }
    }
    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, Some(node_weights))
        .expect("aggregate produces a structurally valid graph")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use single_clustering::network::grouping::{NetworkGrouping, VectorGrouping};

    #[test]
    fn matches_single_clustering_aggregate() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 3000;
        let mut edges = Vec::new();
        for v in 0..n {
            edges.push((v, v, 1.0f64));
            for _ in 0..8 {
                let u = (v + rng.random_range(1..50)) % n;
                edges.push((v, u, rng.random::<f64>()));
            }
        }
        let graph = CSRNetwork::from_edges(n, &edges).unwrap();
        let assign: Vec<usize> = (0..n).map(|v| (v / 7 + v % 3) % 500).collect();
        let grouping = VectorGrouping::from_assignments(&assign);
        let group: Vec<u32> = (0..n).map(|v| grouping.get_group(v) as u32).collect();
        let (ours, theirs) = (aggregate(&graph, &group, grouping.group_count()), graph.aggregate(&grouping));
        assert_eq!(ours.node_count(), theirs.node_count());
        for v in 0..ours.node_count() {
            assert_eq!(ours.neighbors(v).collect::<Vec<_>>(), theirs.neighbors(v).collect::<Vec<_>>(), "node {v}");
            assert_eq!(ours.node_weight(v), theirs.node_weight(v));
        }
        assert_eq!(ours.total_weight(), theirs.total_weight());
    }
}
