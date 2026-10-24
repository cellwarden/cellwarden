Five runs of the final scfast-cpu code on fresh Modal cpu=16 containers, each
`bench/run_py.py --engine scfast-cpu` (warm-up + 2-3 measured repeats). Rounds 1-3 ran at
107ed9d..9c4ef64, rounds 4-5 at fb0b347 (Leiden pool sized by graph size; the only change).
Round 1 has no CPU record; rounds 2-5 record vendor/family/model in env_r*/ (AMD family 25 model 1
= Milan, model 17 = Genoa; Intel family 6 = Xeon). Compute (normalise..UMAP), s:

| Dataset | r1 | r2 | r3 | r4 | r5 | median |
|---|---:|---:|---:|---:|---:|---:|
| PBMC 68k | 2.19 | 1.58 | 2.26 | 1.70 | 1.80 | 1.80 |
| brain 100k | 2.81 | 2.43 | 2.58 | 2.19 | 2.08 | 2.43 |
| brain 500k | 11.06 | 11.86 | 9.48 | 11.27 | 11.43 | 11.27 |
| brain 1.29M | 26.85 | 26.68 | 26.37 | 29.51 | 27.53 | 26.85 |

Leiden step, r1-3 vs r4-5: PBMC 68k 0.73-1.13 -> 0.69-0.72 s; brain 100k 1.13-1.46 -> 0.76-0.83 s.
