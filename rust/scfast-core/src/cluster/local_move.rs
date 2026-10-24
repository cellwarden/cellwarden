//! Batched parallel local moving (RB objective), replacing single-clustering's sequential
//! `local_move` in [`super::leiden_rb`].
//!
//! Same queue discipline as single-clustering: every node starts queued in a shuffled
//! order, and a node that moves re-queues its neighbours outside its new community. Nodes
//! leave the queue in batches; each node of a batch picks its best community in parallel
//! against the state at the start of the batch (single-clustering's gain and tie rules),
//! then the moves are applied in queue order. Batch boundaries depend only on the queue, so
//! the result is the same for any thread count. Before a move is applied its gain is
//! recomputed against the current state (weights to the old and the chosen community,
//! current strengths) and the move is dropped unless it still beats the floor, so every
//! applied move improves the objective and the loop terminates, as the sequential one does.
//!
//! Only the recheck and the bookkeeping are serial. The recheck starts from the proposal's
//! weights and corrects them for the neighbours that moved earlier in the batch (found in
//! parallel), rescanning only when the decision is within rounding of the floor; the
//! neighbours to re-queue are collected and de-duplicated in parallel afterwards. Both give
//! the same decisions and queue as the plain serial loop, so the labels are those of the
//! plain loop.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use single_clustering::community_search::leiden::NeighborWeights;
use single_clustering::network::CSRNetwork;

/// single-clustering's improvement floor, relative to the total edge weight.
const MIN_GAIN_RELATIVE: f64 = 1e-12;
/// single-clustering's move cap per node (a safety valve; moves strictly improve).
const MAX_MOVES_PER_NODE: usize = 100;
/// Batches of n / 16 nodes, at most this many. Smaller batches mean more fork-join
/// barriers: on 500k cells n / 64 (capped at 4096) took 2.9 s against 2.6 s.
const MAX_BATCH: usize = 65_536;
/// Applied moves per parallel task when collecting the neighbours to re-queue.
const WAKE_CHUNK: usize = 512;
/// Proposals per parallel task when collecting conflicting neighbours.
const CONFLICT_CHUNK: usize = 1024;
/// Relative distance from the floor below which a move decided on corrected weights is
/// re-decided on a rescan (the corrections are exact up to a few ulps).
const ROUNDING_MARGIN: f64 = 1e-9;
/// Stands for "a fresh empty community" in a proposal.
const EMPTY: u32 = u32::MAX;

thread_local! {
    static WEIGHTS: RefCell<NeighborWeights> = RefCell::new(NeighborWeights::with_capacity(0));
}

/// Moves nodes of `graph` between the communities of `start` (community of each node) until
/// no single move improves the RB objective by more than the floor. Returns the new
/// membership.
pub(super) fn local_move(graph: &CSRNetwork, start: &[usize], resolution: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let n = graph.node_count();
    let four_m = 4.0 * graph.total_weight();
    let threshold = MIN_GAIN_RELATIVE * graph.total_weight().abs().max(1.0);
    let strength: Vec<f64> = (0..n).into_par_iter().map(|v| graph.strength(v)).collect();
    let self_loop: Vec<f64> = (0..n).into_par_iter().map(|v| graph.self_loop_weight(v)).collect();
    let mut membership: Vec<u32> = start.iter().map(|&c| c as u32).collect();
    let slots = start.iter().copied().max().map_or(0, |m| m + 1).max(n) + 1;
    let mut kc = vec![0f64; slots];
    let mut size = vec![0u32; slots];
    for (v, &c) in membership.iter().enumerate() {
        kc[c as usize] += strength[v];
        size[c as usize] += 1;
    }
    let mut free: Vec<u32> = (0..slots as u32).rev().filter(|&c| size[c as usize] == 0).collect();

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<u32> = order.into();
    let mut queued = vec![true; n];
    let batch_len = (n / 16).clamp(1, MAX_BATCH);
    let mut moves = 0usize;
    let budget = n.saturating_mul(MAX_MOVES_PER_NODE);
    if four_m == 0.0 {
        return membership.iter().map(|&c| c as usize).collect();
    }

    let gain = |w: f64, kv: f64, sl: f64, k_c: f64| (w + sl) - resolution * kv * (2.0 * k_c + kv) / four_m;
    let mut batch = Vec::with_capacity(batch_len);
    let mut proposals = Vec::with_capacity(batch_len);
    // per batch: the applied moves (node, from) in order, each moved node's 1-based position
    // in that list, and each proposing node's 1-based position in the batch
    let mut applied: Vec<(u32, u32)> = Vec::new();
    let mut moved_at = vec![0u32; n];
    let mut proposed_at = vec![0u32; n];
    let first_wake: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(u32::MAX)).collect();
    while !queue.is_empty() && moves < budget {
        batch.clear();
        batch.extend(queue.drain(..batch_len.min(queue.len())));
        for &v in &batch {
            queued[v as usize] = false;
        }
        batch
            .par_iter()
            .map(|&v| {
                let v = v as usize;
                WEIGHTS.with_borrow_mut(|nw| {
                    nw.ensure_capacity(slots);
                    nw.collect(graph, &membership, v);
                    let (kv, sl) = (strength[v], self_loop[v]);
                    let old = membership[v];
                    let w_old = nw.weight_to(old as usize);
                    let base = gain(w_old, kv, sl, kc[old as usize] - kv);
                    let (mut best, mut best_gain) = (old, base);
                    for &c in nw.touched() {
                        if c as u32 == old {
                            continue;
                        }
                        let g = gain(nw.weight_to(c), kv, sl, kc[c]);
                        if g > best_gain || (g == best_gain && (c as u32) < best) {
                            (best, best_gain) = (c as u32, g);
                        }
                    }
                    if size[old as usize] > 1 {
                        let g = gain(0.0, kv, sl, 0.0);
                        if g > best_gain {
                            (best, best_gain) = (EMPTY, g);
                        }
                    }
                    let w_best = if best == EMPTY { 0.0 } else { nw.weight_to(best as usize) };
                    (best != old && best_gain - base > threshold).then_some((best, w_old, w_best))
                })
            })
            .collect_into_vec(&mut proposals);

        // per proposal, in batch order: the neighbours that proposed earlier in the batch a
        // move that may enter or leave its old or target community (a move to a fresh
        // community may land anywhere), with their edge weights
        for (i, (&v, p)) in batch.iter().zip(&proposals).enumerate() {
            if p.is_some() {
                proposed_at[v as usize] = i as u32 + 1;
            }
        }
        let conflicts: Vec<Vec<(u32, u32, f64)>> = proposals
            .par_chunks(CONFLICT_CHUNK)
            .enumerate()
            .map(|(chunk, props)| {
                let mut out = Vec::new();
                for (j, p) in props.iter().enumerate() {
                    let Some((target, _, _)) = *p else { continue };
                    let i = chunk * CONFLICT_CHUNK + j;
                    let v = batch[i] as usize;
                    let old = membership[v];
                    for (u, w) in graph.neighbors(v) {
                        let at = proposed_at[u] as usize;
                        if at == 0 || at > i {
                            continue;
                        }
                        let (u_target, _, _) = proposals[at - 1].expect("proposed_at marks proposals");
                        let u_old = membership[u];
                        if u_target == EMPTY || u_old == old || u_old == target || u_target == old || u_target == target {
                            out.push((i as u32, u as u32, w));
                        }
                    }
                }
                out
            })
            .collect();
        for &v in &batch {
            proposed_at[v as usize] = 0;
        }

        applied.clear();
        let mut conflicts = conflicts.iter().flatten().peekable();
        for (i, (&v, &p)) in batch.iter().zip(&proposals).enumerate() {
            let Some((mut target, mut w_old, mut w_target)) = p else { continue };
            let v = v as usize;
            let old = membership[v];
            let (kv, sl) = (strength[v], self_loop[v]);
            // The proposal's weights, corrected for the conflicting neighbours that have
            // moved since. Unchanged, they are the ones a rescan would give bit for bit;
            // corrected, they can differ from a rescan by rounding, so a decision within a
            // rounding margin of the floor is made on a rescan.
            let mut corrected = false;
            let mut magnitude = w_old + w_target;
            while let Some(&(_, u, w)) = conflicts.next_if(|c| c.0 as usize == i) {
                let at = moved_at[u as usize] as usize;
                if at == 0 {
                    continue;
                }
                let (from, to) = (applied[at - 1].1, membership[u as usize]);
                for (c, weight) in [(old, &mut w_old), (target, &mut w_target)] {
                    if from == c {
                        *weight -= w;
                        corrected = true;
                    }
                    if to == c {
                        *weight += w;
                        corrected = true;
                    }
                }
                magnitude += w;
            }
            let k_target = if target == EMPTY { 0.0 } else { kc[target as usize] };
            let (mut g_target, mut g_old) = (gain(w_target, kv, sl, k_target), gain(w_old, kv, sl, kc[old as usize] - kv));
            if corrected && (g_target - g_old - threshold).abs() <= ROUNDING_MARGIN * (magnitude + sl + g_target.abs() + g_old.abs()) {
                (w_old, w_target) = (0.0, 0.0);
                for (u, w) in graph.neighbors(v) {
                    if u != v {
                        let c = membership[u];
                        if c == old {
                            w_old += w;
                        } else if c == target {
                            w_target += w;
                        }
                    }
                }
                (g_target, g_old) = (gain(w_target, kv, sl, k_target), gain(w_old, kv, sl, kc[old as usize] - kv));
            }
            if g_target - g_old <= threshold {
                continue;
            }
            if target == EMPTY {
                if size[old as usize] == 1 {
                    continue;
                }
                target = loop {
                    let c = free.pop().expect("a non-singleton community leaves an empty slot");
                    if size[c as usize] == 0 {
                        break c;
                    }
                };
            }
            membership[v] = target;
            kc[old as usize] -= strength[v];
            kc[target as usize] += strength[v];
            size[old as usize] -= 1;
            size[target as usize] += 1;
            if size[old as usize] == 0 {
                free.push(old);
            }
            moves += 1;
            applied.push((v as u32, old));
            moved_at[v] = applied.len() as u32;
        }

        // Neighbours to re-queue, per applied move in order: those outside the mover's new
        // community at the time of its move (a neighbour that moved later in the batch was
        // still in its old community then) and not already queued.
        let wake: Vec<Vec<u32>> = applied
            .par_chunks(WAKE_CHUNK)
            .enumerate()
            .map(|(chunk, moves)| {
                let mut out = Vec::new();
                for (i, &(v, _)) in moves.iter().enumerate() {
                    let pos = (chunk * WAKE_CHUNK + i + 1) as u32;
                    let v = v as usize;
                    let target = membership[v];
                    for (u, _) in graph.neighbors(v) {
                        if u == v || queued[u] {
                            continue;
                        }
                        let at = moved_at[u];
                        let c = if at > pos { applied[at as usize - 1].1 } else { membership[u] };
                        if c != target {
                            out.push(u as u32);
                        }
                    }
                }
                out
            })
            .collect();
        // first occurrence of each neighbour wins, as if they were queued one by one
        let mut offsets = Vec::with_capacity(wake.len());
        let mut total = 0u32;
        for w in &wake {
            offsets.push(total);
            total += w.len() as u32;
        }
        wake.par_iter().zip(&offsets).for_each(|(w, &off)| {
            for (j, &u) in w.iter().enumerate() {
                first_wake[u as usize].fetch_min(off + j as u32, Ordering::Relaxed);
            }
        });
        let woken: Vec<Vec<u32>> = wake
            .par_iter()
            .zip(&offsets)
            .map(|(w, &off)| {
                let mine = w.iter().enumerate().filter(|&(j, &u)| first_wake[u as usize].load(Ordering::Relaxed) == off + j as u32);
                mine.map(|(_, &u)| u).collect()
            })
            .collect();
        woken.par_iter().flatten().for_each(|&u| first_wake[u as usize].store(u32::MAX, Ordering::Relaxed));
        for w in &woken {
            for &u in w {
                queued[u as usize] = true;
            }
            queue.extend(w);
        }
        for &(v, _) in &applied {
            moved_at[v as usize] = 0;
        }
    }
    membership.iter().map(|&c| c as usize).collect()
}
