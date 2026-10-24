//! Exact kNN with a k-d tree, for low-dimensional embeddings (the 10 leading PCs).
//!
//! The tree is implicit and balanced: node `v` has children `2v + 1` and `2v + 2`, and
//! splits its point range at the middle along the dimension of largest spread, so all
//! leaves sit at one depth and hold `LEAF_SIZE / 2 ..= LEAF_SIZE` points. Each node keeps
//! its bounding box. Points are stored in tree order (f32), each leaf as one contiguous
//! block laid out dimension-major, so the leaf scan vectorises.
//!
//! Every distance is `exact_d2` (f64, same summation order), and a node is skipped only
//! when its box lower bound is strictly greater than the current (k - 1)-th distance. The
//! box bound sums, in the same order, terms that are never larger than the point's terms,
//! and rounding is monotone, so it never exceeds `exact_d2` of any point in the box: the
//! result is the brute-force answer, bit for bit, with the same index tie-break.

use super::{cmp_neighbour, write_row};
use crate::types::{Knn, RowMajor};
use rayon::prelude::*;

const LEAF_SIZE: usize = 64;
/// Queries handled per parallel task, in tree order so neighbouring queries share cache.
const QUERY_CHUNK: usize = 256;

struct Tree {
    d: usize,
    depth: usize,
    /// Original index of the point at each tree position.
    perm: Vec<u32>,
    /// Points in tree order (f32, as given; widened to f64 when used); leaf `[s, e)` is the block `s * d .. e * d`, holding
    /// dimension `t` of its points at `t * (e - s) ..`.
    pts: Vec<f32>,
    /// Per node: d lower then d upper box bounds.
    boxes: Vec<f32>,
}

impl Tree {
    fn build(x: &RowMajor) -> Self {
        let (n, d) = (x.n_rows, x.n_cols);
        let mut depth = 0;
        while n >> depth > LEAF_SIZE {
            depth += 1;
        }
        let mut perm: Vec<u32> = (0..n as u32).collect();
        split(x, &mut perm, depth);
        let first_leaf = (1usize << depth) - 1;
        let leaves: Vec<(usize, usize)> = (0..1usize << depth).map(|l| range(n, depth, first_leaf + l)).collect();
        let mut pts = vec![0f32; n * d];
        let mut blocks = Vec::with_capacity(leaves.len());
        let mut rest = pts.as_mut_slice();
        for &(s, e) in &leaves {
            let (head, tail) = rest.split_at_mut((e - s) * d);
            blocks.push((s, e, head));
            rest = tail;
        }
        blocks.into_par_iter().for_each(|(s, e, blk)| {
            let len = e - s;
            for (j, &i) in perm[s..e].iter().enumerate() {
                for (t, &v) in x.row(i as usize).iter().enumerate() {
                    blk[t * len + j] = v;
                }
            }
        });
        let n_nodes = (1usize << (depth + 1)) - 1;
        let mut boxes = vec![0f32; n_nodes * 2 * d];
        boxes[first_leaf * 2 * d..].par_chunks_mut(2 * d).zip(&leaves).for_each(|(b, &(s, e))| {
            let len = e - s;
            let blk = &pts[s * d..e * d];
            for t in 0..d {
                let col = &blk[t * len..(t + 1) * len];
                b[t] = col.iter().copied().fold(f32::INFINITY, f32::min);
                b[d + t] = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
        });
        for v in (0..first_leaf).rev() {
            let (parent, kids) = boxes.split_at_mut((2 * v + 1) * 2 * d);
            let (l, r) = kids.split_at(2 * d);
            let b = &mut parent[v * 2 * d..];
            for t in 0..d {
                b[t] = l[t].min(r[t]);
                b[d + t] = l[d + t].max(r[d + t]);
            }
        }
        Self { d, depth, perm, pts, boxes }
    }

    /// Coordinates of the point at tree position `pos`, gathered from its leaf block.
    fn point<'a>(&self, pos: usize, q: &'a mut [f64]) -> &'a [f64] {
        let leaf = (1usize << self.depth) - 1 + leaf_of(self.perm.len(), self.depth, pos);
        let (s, e) = range(self.perm.len(), self.depth, leaf);
        for (t, v) in q.iter_mut().enumerate() {
            *v = self.pts[s * self.d + t * (e - s) + pos - s] as f64;
        }
        q
    }

    /// Squared distance from `q` to node `v`'s box, never above `exact_d2` of a point inside.
    #[inline]
    fn box_d2(&self, v: usize, q: &[f64]) -> f64 {
        let b = &self.boxes[v * 2 * self.d..(v + 1) * 2 * self.d];
        let (lo, hi) = b.split_at(self.d);
        let mut s = 0.0;
        for t in 0..self.d {
            let g = (lo[t] as f64 - q[t]).max(q[t] - hi[t] as f64).max(0.0);
            s += g * g;
        }
        s
    }

    /// Best-first search: nodes are opened in increasing box distance, each one followed
    /// down its nearer side, the farther sides queued. Stops when the nearest queued box is
    /// beyond the current (k - 1)-th squared distance times `shrink` (1 for exact search).
    fn search(&self, q: &[f64], self_pos: usize, shrink: f64, best: &mut Best, queue: &mut Vec<Pending>, scratch: &mut [f64]) {
        queue.clear();
        queue.push(Pending { lb: 0.0, v: 0, s: 0, e: self.perm.len() as u32, level: 0 });
        while let Some(Pending { lb, mut v, s, e, mut level }) = pop_min(queue) {
            if lb > best.worst() * shrink {
                break;
            }
            let (mut s, mut e) = (s as usize, e as usize);
            while level < self.depth as u32 {
                let m = s + (e - s) / 2;
                let (l, r) = (2 * v + 1, 2 * v + 2);
                let (dl, dr) = (self.box_d2(l as usize, q), self.box_d2(r as usize, q));
                let (near, far, dn, df, ns, ne, fs, fe) =
                    if dl <= dr { (l, r, dl, dr, s, m, m, e) } else { (r, l, dr, dl, m, e, s, m) };
                if df <= best.worst() * shrink {
                    push(queue, Pending { lb: df, v: far, s: fs as u32, e: fe as u32, level: level + 1 });
                }
                if dn > best.worst() * shrink {
                    level = u32::MAX;
                    break;
                }
                (v, s, e, level) = (near, ns, ne, level + 1);
            }
            if level == u32::MAX {
                continue;
            }
            let len = e - s;
            let acc = &mut scratch[..len];
            leaf_d2(&self.pts[s * self.d..e * self.d], q, acc);
            let mut worst = best.worst();
            for (j, &d2) in acc.iter().enumerate() {
                if d2 <= worst && s + j != self_pos {
                    best.insert((d2, self.perm[s + j]));
                    worst = best.worst();
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Pending {
    lb: f64,
    v: u32,
    s: u32,
    e: u32,
    level: u32,
}

/// Binary min-heap on `lb` kept in a plain Vec (reused across queries).
fn push(h: &mut Vec<Pending>, x: Pending) {
    h.push(x);
    let mut i = h.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if h[p].lb <= x.lb {
            break;
        }
        h[i] = h[p];
        i = p;
    }
    h[i] = x;
}

fn pop_min(h: &mut Vec<Pending>) -> Option<Pending> {
    let top = *h.first()?;
    let last = h.pop().unwrap();
    let n = h.len();
    if n > 0 {
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let c = if l + 1 < n && h[l + 1].lb < h[l].lb { l + 1 } else { l };
            if h[c].lb >= last.lb {
                break;
            }
            h[i] = h[c];
            i = c;
        }
        h[i] = last;
    }
    Some(top)
}

/// Squared distances from `q` to the points of a dimension-major leaf block. The first
/// dimension initialises `acc` (`0 + x == x`), so the sums match `exact_d2` bit for bit.
#[inline]
fn leaf_d2(blk: &[f32], q: &[f64], acc: &mut [f64]) {
    let len = acc.len();
    let mut cols = blk.chunks_exact(len).zip(q);
    if let Some((col, &q0)) = cols.next() {
        for (a, &v) in acc.iter_mut().zip(col) {
            let g = q0 - v as f64;
            *a = g * g;
        }
    }
    for (col, &qt) in cols {
        for (a, &v) in acc.iter_mut().zip(col) {
            let g = qt - v as f64;
            *a += g * g;
        }
    }
}

/// Point range of node `v` in an implicit tree of the given depth over n points.
fn range(n: usize, depth: usize, v: usize) -> (usize, usize) {
    let level = (v + 1).ilog2() as usize;
    let mut idx = v + 1 - (1 << level);
    let (mut s, mut e) = (0, n);
    for l in (0..level).rev() {
        let m = s + (e - s) / 2;
        if idx >> l & 1 == 0 {
            e = m;
        } else {
            s = m;
        }
        idx &= (1 << l) - 1;
    }
    debug_assert!(level <= depth);
    (s, e)
}

/// Index (0-based, left to right) of the leaf holding tree position `pos`.
fn leaf_of(n: usize, depth: usize, pos: usize) -> usize {
    let (mut s, mut e, mut leaf) = (0, n, 0);
    for _ in 0..depth {
        let m = s + (e - s) / 2;
        leaf <<= 1;
        if pos < m {
            e = m;
        } else {
            s = m;
            leaf |= 1;
        }
    }
    leaf
}

/// Reorders `perm` so every node's points are contiguous, splitting at the middle along the
/// dimension of largest spread.
fn split(x: &RowMajor, perm: &mut [u32], levels: usize) {
    if levels == 0 {
        return;
    }
    let d = x.n_cols;
    let mut lo = vec![f32::INFINITY; d];
    let mut hi = vec![f32::NEG_INFINITY; d];
    for &i in perm.iter() {
        for (t, &v) in x.row(i as usize).iter().enumerate() {
            lo[t] = lo[t].min(v);
            hi[t] = hi[t].max(v);
        }
    }
    let dim = (0..d).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
    let m = perm.len() / 2;
    perm.select_nth_unstable_by(m, |&a, &b| x.row(a as usize)[dim].total_cmp(&x.row(b as usize)[dim]));
    let (l, r) = perm.split_at_mut(m);
    if l.len() > 4096 {
        rayon::join(|| split(x, l, levels - 1), || split(x, r, levels - 1));
    } else {
        split(x, l, levels - 1);
        split(x, r, levels - 1);
    }
}

/// The best `cap` candidates so far, sorted by `cmp_neighbour`.
struct Best {
    items: Vec<(f64, u32)>,
    cap: usize,
}

impl Best {
    #[inline]
    fn worst(&self) -> f64 {
        if self.items.len() < self.cap { f64::INFINITY } else { self.items[self.cap - 1].0 }
    }

    #[inline]
    fn insert(&mut self, c: (f64, u32)) {
        if self.items.len() == self.cap {
            if cmp_neighbour(&c, &self.items[self.cap - 1]).is_ge() {
                return;
            }
            self.items.pop();
        }
        self.items.push(c);
        let mut i = self.items.len() - 1;
        while i > 0 && cmp_neighbour(&self.items[i - 1], &c).is_gt() {
            self.items[i] = self.items[i - 1];
            i -= 1;
        }
        self.items[i] = c;
    }
}

/// `eps = 0` gives the exact answer. With `eps > 0` a node is skipped when its box is
/// farther than the current (k - 1)-th distance divided by `1 + eps`, so each reported
/// neighbour is within `1 + eps` times the distance of the true one of the same rank.
pub(super) fn kdtree_knn(x: &RowMajor, k: usize, eps: f64) -> Knn {
    let shrink = 1.0 / ((1.0 + eps) * (1.0 + eps));
    let (n, d) = (x.n_rows, x.n_cols);
    let tree = Tree::build(x);
    let mut tidx = vec![0u32; n * k];
    let mut tdist = vec![0f32; n * k];
    tidx.par_chunks_mut(QUERY_CHUNK * k).zip(tdist.par_chunks_mut(QUERY_CHUNK * k)).enumerate().for_each_init(
        || (Best { items: Vec::with_capacity(k), cap: k - 1 }, vec![0f64; d], Vec::new(), vec![0f64; LEAF_SIZE + 1]),
        |(best, q, queue, scratch), (c, (ib, db))| {
            for (t, (irow, drow)) in ib.chunks_exact_mut(k).zip(db.chunks_exact_mut(k)).enumerate() {
                let pos = c * QUERY_CHUNK + t;
                best.items.clear();
                tree.search(tree.point(pos, q), pos, shrink, best, queue, scratch);
                write_row(tree.perm[pos] as usize, &best.items, irow, drow);
            }
        },
    );
    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];
    let mut inv = vec![0u32; n];
    for (pos, &i) in tree.perm.iter().enumerate() {
        inv[i as usize] = pos as u32;
    }
    indices.par_chunks_mut(k).zip(distances.par_chunks_mut(k)).zip(&inv).for_each(|((irow, drow), &pos)| {
        let p = pos as usize;
        irow.copy_from_slice(&tidx[p * k..(p + 1) * k]);
        drow.copy_from_slice(&tdist[p * k..(p + 1) * k]);
    });
    Knn { n, k, indices, distances }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_tile_each_level() {
        for n in [1, 33, 64, 1000, 4097] {
            let mut depth = 0;
            while n >> depth > LEAF_SIZE {
                depth += 1;
            }
            for level in 0..=depth {
                let first = (1usize << level) - 1;
                let mut next = 0;
                for v in first..first + (1 << level) {
                    let (s, e) = range(n, depth, v);
                    assert_eq!(s, next);
                    next = e;
                }
                assert_eq!(next, n);
            }
        }
    }
}
