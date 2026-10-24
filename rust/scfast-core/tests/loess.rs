//! `preprocess::loess_interpolate` against R's default `loess(y ~ x, span, degree = 2)`
//! (fixture and the R script that made it in tests/data/).

use scfast_core::preprocess::{loess_direct, loess_interpolate};

#[test]
fn loess_interpolate_matches_r() {
    let text = include_str!("data/loess_interpolate.tsv");
    let rows: Vec<[f64; 5]> = text
        .lines()
        .skip(2)
        .map(|l| {
            let v: Vec<f64> = l.split('\t').map(|s| s.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4]]
        })
        .collect();
    for case in [1.0, 2.0] {
        let r: Vec<&[f64; 5]> = rows.iter().filter(|r| r[0] == case).collect();
        let span = r[0][1];
        let (x, y): (Vec<f64>, Vec<f64>) = r.iter().map(|r| (r[2], r[3])).unzip();
        let fit = loess_interpolate(&x, &y, span).unwrap();
        let err = fit.iter().zip(&r).map(|(f, r)| (f - r[4]).abs()).fold(0.0, f64::max);
        assert!(err < 1e-10, "case {case}: max |fit - R| = {err:e}");
        // The surfaces differ, so the fixture tells them apart.
        let direct = loess_direct(&x, &y, span).unwrap();
        let gap = direct.iter().zip(&r).map(|(f, r)| (f - r[4]).abs()).fold(0.0, f64::max);
        assert!(gap > 1e-6, "case {case}: direct is within {gap:e} of R's interpolated fit");
    }
}
