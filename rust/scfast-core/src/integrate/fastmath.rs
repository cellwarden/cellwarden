//! `exp` and `ln` for the soft-assignment loops: about three times the throughput of
//! glibc's (2.4 against 8 ns per `exp` on AMD EPYC, where the distance GEMM costs ~2 ns
//! per entry of R). `exp` is branch-free, so the compiler vectorises the loops that map it
//! over a column.
//!
//! Both reduce to a polynomial on a short interval: `exp(x) = 2^n exp(r)` with
//! `|r| <= ln 2 / 2` (degree-12 Taylor), `ln(x) = e ln 2 + 2 atanh(s)` with
//! `s = (m - 1) / (m + 1)`, `m` in `[sqrt(1/2), sqrt(2))` (series to `s^23`). Relative
//! error below 1e-15 on the ranges used here (tested against std); results are the same on
//! every thread, so determinism is unaffected.

const LN2_HI: f64 = 0.693_147_180_369_123_8;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
/// 1.5 * 2^52: adding and subtracting it rounds to the nearest integer.
const ROUND: f64 = 6_755_399_441_055_744.0;

/// `e^x` for finite `x <= 0`; arguments below -708 are clamped there (`e^-708` ~ 3e-308,
/// against the 1 of the largest entry the soft assignment normalises by).
#[inline(always)]
pub(super) fn exp_neg(x: f64) -> f64 {
    let x = x.max(-708.0);
    let t = x * std::f64::consts::LOG2_E + ROUND;
    let n = t - ROUND;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // Taylor to r^12, split in two Horner chains to shorten the dependency chain
    let hi = (((r / 479_001_600.0 + 1.0 / 39_916_800.0) * r + 1.0 / 3_628_800.0) * r + 1.0 / 362_880.0) * r + 1.0 / 40_320.0;
    let mid = ((hi * r + 1.0 / 5_040.0) * r + 1.0 / 720.0) * r + 1.0 / 120.0;
    let p = (((mid * r + 1.0 / 24.0) * r + 1.0 / 6.0) * r + 0.5) * (r * r) + r + 1.0;
    // 2^n from the integer n held in the low mantissa bits of t (n >= -1022 here)
    let bits = ((t.to_bits() as i64 - ROUND.to_bits() as i64 + 1023) << 52) as u64;
    p * f64::from_bits(bits)
}

/// Natural logarithm of a positive, finite, normal `x`.
#[inline(always)]
pub(super) fn ln_pos(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    let ef = e as f64;
    ef * LN2_HI + (2.0 * s * p + ef * LN2_LO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        let mut worst = 0f64;
        for i in 0..=200_000 {
            let x = -745.0 * i as f64 / 200_000.0;
            let (a, b) = (exp_neg(x), x.exp());
            if x > -708.0 {
                worst = worst.max(((a - b) / b).abs());
            } else {
                assert!(a < 1e-307, "{x}: {a}");
            }
        }
        assert!(worst < 1e-15, "max relative error {worst:e}");
        assert_eq!(exp_neg(0.0), 1.0);
    }

    #[test]
    fn ln_matches_std() {
        let mut worst = 0f64;
        for i in 1..=200_000 {
            for x in [i as f64 / 200_000.0, i as f64 * 1e-3, 1e-300 * i as f64, 1.0 + i as f64 * 1e-9] {
                let (a, b) = (ln_pos(x), x.ln());
                worst = worst.max(if b.abs() > 1e-3 { ((a - b) / b).abs() } else { (a - b).abs() });
            }
        }
        assert!(worst < 1e-15, "max error {worst:e}");
        assert_eq!(ln_pos(1.0), 0.0);
    }
}
