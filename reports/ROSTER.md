# Sweep roster

Cost of each sweep before it is launched, measured on the RTX 4050 laptop (bf16, eval every 10 steps on 512 held-out windows).

| sweep | config | cells | steps | est. wall | status |
|---|---|---|---|---|---|
| smoke | `configs/experiments/smoke.yaml` | 1 scale × 3 seeds × 1 fork × 4 strategies | 2,340 | 5 min (measured 4m51s) | done, pipeline check only |
| seed sweep | `configs/experiments/shakespeare_scales.yaml` | 3 scales × 5 seeds × 4 forks × 13 strategies = 780 migrations + 15 controls | ≈ 264,000 | ≈ 12–14 h | running |

Per-step cost from the smoke run is about 0.1 s for gpt-xs including evaluation. gpt-s and gpt-m are about 1.5× and 2.5× that. Warm strategies add 20 gradient passes per migration.

Step times recorded in these manifests come from a laptop that is also running other work (CPU tests, compiler and MPI benchmarks), so they are not used for the cost model. Timing for `policy.py` comes from dedicated runs, see `reports/timing/`.
