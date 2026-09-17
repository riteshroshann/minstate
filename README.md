# Minimum Viable State for Elastic Distributed Training

**High-Performance and Cloud Computing (HPCC) — Semester V**

> How much optimizer state does a distributed training job actually have to *carry* when it migrates between machines, and how much of it can be *rebuilt* at the destination instead?

**Status: design and documentation complete; no experiments have been run yet.** Every table marked `TBD` is a placeholder awaiting measured data. Nothing in this document is an illustrative or invented result.

---

## 1. Motivation

Preemptible ("spot") instances are offered at a 70–80% discount against on-demand pricing, with a reclamation notice on the order of seconds. Using them for training therefore requires jobs that can *relocate quickly*. Relocation means moving the training state to a new machine — and for a large model, that state is large.

### 1.1 Where the bytes go

For a model with `N` parameters trained with Adam, the state resident on the accelerators is:

| Component | Bytes per parameter | Necessary at destination? |
|---|---|---|
| Model weights (fp32 master) | 4 | **Yes** |
| Adam first moment `m` | 4 | Under investigation |
| Adam second moment `v` | 4 | Under investigation |
| **Core optimizer state** | **12** | |
| Gradients (fp32) | 4 | No — recomputed next step |
| bf16 compute copy of weights | 2 | No — derived from master |
| EMA / averaged weights, if used | 4 | Only if evaluation needs it |
| Framework buffers, comm scratch | ~2–6 | No |
| **Typical full snapshot** | **~18–28** | |

At 7B parameters, the 12 B/param core is **84 GB**; a full framework snapshot at ~28 B/param is **≈196 GB**, which is where the commonly quoted "~200 GB to migrate a 7B job" figure comes from. The reducible part this project targets is the **8 B/param of optimizer moments — 56 GB at 7B, two thirds of the core state.**

*(Arithmetic extrapolation from per-parameter accounting, not a measurement on this hardware. See §7.)*

### 1.2 Why this is not just a cost question

Egress charges and accelerator idle time frequently exceed the spot discount — but there is a harder constraint underneath. The payload has to *drain before the instance dies*. Preemption notice windows are roughly 30 s (GCP Spot, Azure Spot) to 120 s (AWS EC2 Spot).

Drain time at 7B, assuming the payload is moved at the full link rate:

| Payload | Size @ 7B | Drain @ 10 Gb/s | Drain @ 25 Gb/s | Fits a 30 s notice? |
|---|---|---|---|---|
| Full snapshot (28 B/param) | 196 GB | 157 s | 63 s | No |
| Core Adam state (12 B/param) | 84 GB | 67 s | 27 s | Only at 25 Gb/s |
| Weights only, fp32 (4 B/param) | 28 GB | 22 s | 9 s | **Yes** |
| Weights only, bf16 (2 B/param) | 14 GB | 11 s | 4.5 s | **Yes** |

So the difference between carrying the moments and rebuilding them is not merely a cost optimization — at 10 Gb/s it is the difference between a migration that completes inside the notice window and one that does not. That reframes the question: **not "how do we migrate less often" (the existing approach) but "how small can the payload be made".**

### 1.3 The hypothesis worth testing

Model weights are necessary at the destination — nothing reconstructs them. The optimizer moments are not obviously necessary. Both are exponential moving averages over recent gradients, and are in principle recoverable from a short warm-up on the receiving machine.

Crucially, the two moments have very different memory horizons. With the standard `β1 = 0.9`, `β2 = 0.999`:

- `m` has an effective horizon of `1/(1-β1) ≈ 10` steps.
- `v` has an effective horizon of `1/(1-β2) ≈ 1000` steps.

If rebuild cost tracks horizon, then **`m` is nearly free to reconstruct and `v` is not** — which predicts that the useful payload reduction is "carry `v`, drop `m`", the opposite of the intuitive ordering, and that a cheap *approximation* of `v` may capture most of its value. That is the core question this project measures rather than assumes.

---

## 2. Research questions

- **RQ1** — What is the tradeoff curve between bytes transferred at migration and training steps lost to recovery?
- **RQ2** — Is the tradeoff asymmetric between the first and second moments, as the horizon argument predicts?
- **RQ3** — Can a low-rank (row/column) approximation of `v`, costing ~0.3% of `v`'s bytes, recover most of the benefit of shipping `v` in full?
- **RQ4** — Does the cost of dropping state depend on *where* in the training schedule the migration happens?
- **RQ5** — Given the measured curve, when does a spot-price differential actually justify migrating, and which payload should be chosen?

### Hypotheses

| | Hypothesis | Rationale |
|---|---|---|
| **H1** | Dropping `m` costs ≈10 steps; dropping `v` costs orders of magnitude more. | EMA horizons, §1.3. |
| **H2** | `weights + rank-1 factored v` sits near the `full` arm's recovery at near weights-only bytes, dominating the Pareto frontier. | Adafactor trains from scratch using only a factored second moment. |
| **H3** | Recovery cost varies with migration point in the schedule. | Open which direction: late-stage gradients are smaller and the loss surface more anisotropic (argues *worse*), but late-stage `v` is more stationary (argues *better*). |
| **H4** | Embedding and output-projection moments dominate the residual gap. | Their gradients are sparse — only sampled tokens update — so their `v` encodes long-horizon information a short warm-up cannot rebuild. |
| **H5** | A frozen warm-up converts bytes into compute at a computable exchange rate, with a crossover bandwidth below which rebuilding beats shipping. | Warm-up cost is bounded and predictable; a cold resume's damage is not. |

---

## 3. Method

### 3.1 The ablation

Run an uninterrupted **control run**. Then re-run it, interrupting at step `T`, carrying only a subset of the state across the boundary, and resuming. Measure how long the resumed run takes to rejoin the control's loss trajectory.

**Arms** — what crosses the boundary (`theta` = weights):

| Arm | Carried | Core B/param | vs. `full` |
|---|---|---|---|
| `full` | theta, m, v | 12 | 100% |
| `w_v` | theta, v | 8 | 67% |
| `w_m` | theta, m | 8 | 67% |
| `w_only` | theta | 4 | 33% |
| `w_factored_v` | theta + rank-1 row/col factors of v | ≈ 4.01 | ≈ 33% |
| `w_lowrank_v` | theta + rank-r truncated factors of v | 4 + 4r(d₁+d₂)/(d₁d₂) | 33–35% |
| `w_emb_v` | theta + full v for embedding/output tensors only | ≈ 4 + emb. share | ~35–45% |
| `cold_reset` | theta, moments zeroed *and* Adam's step counter reset | 4 | 33% (floor) |

Adam's scalar step counter `t` is carried in every arm except `cold_reset` — it is ~8 bytes, and dropping it breaks bias correction in a way that has nothing to do with the moments.

**Protocols** — what the destination does on arrival:

- `cold` — resume immediately with whatever arrived.
- `frozen_warmup(W)` — run `W` steps accumulating gradients into `m` and `v` while applying **no** parameter update, then resume. Pays for reconstruction in compute; risks no divergence.
- `lr_rewarmup(W)` — resume immediately with the learning rate ramped linearly from 0 over `W` steps.

**Axes:** model scale × migration point (`early` ≈5%, `mid` ≈40%, `late` ≈85% of schedule) × arm × protocol.

### 3.2 Why a cold resume hurts at all

With `m = v = 0` but `t` preserved, the bias-correction divisors are ≈1, so the first update is ≈0. On the second step `m ≈ (1-β1)g` and `v ≈ (1-β2)g²`, giving an update of roughly

```
(1-β1)/sqrt(1-β2)  ≈  0.1 / 0.0316  ≈  3.2×
```

the normal step size, aimed along a single noisy gradient. The transient is an **effective-learning-rate spike with a badly conditioned preconditioner**, not "forgetting". This is why `frozen_warmup` is expected to help: it rebuilds the preconditioner before any parameter moves.

### 3.3 Metrics

**Seed band first.** Steps-to-recovery is meaningless without a noise floor, so ≥5 control runs differing only in seed are run before any ablation. `eps` = 95th percentile of pairwise loss differences across seeds over the recovery window. A migrated run counts as recovered only once it is inside that band, and `eps` is reported next to every result.

**Steps-to-Recovery (StR)** — the first step `k` after which the migrated run stays within `eps` of the mean control trajectory for `K = 50` consecutive steps. Evaluated on a fixed held-out batch set, not on training loss. If recovery does not occur within the window `W`, the value is reported censored as `>W`.

**Regret area** — `Σ max(0, L_mig − L_ctrl)` over the window. More robust than StR, and reported alongside it; where the two disagree, the disagreement distinguishes a deep-but-brief transient from a shallow-but-long one.

**Payload bytes** — measured from the actual serialized bundle, never derived from parameter counts.

### 3.4 What makes the comparison valid

Three invariants, enforced in code and covered by tests (details in [CLAUDE.md](CLAUDE.md) §6):

1. **Data order is a pure function of the global step index**, so the migrated run sees exactly the batches the control saw after `T`.
2. **RNG is step-keyed**, so dropout masks match and StR is not partly measuring dropout noise.
3. **Exactly one thing differs** between control and migrated run: what crossed the boundary, and what the destination did with it.

### 3.5 Model scales

Sized for a 6 GB laptop GPU (§7). Vocabulary is held at 8192 BPE tokens so that non-embedding parameters dominate, since the science concerns transformer weight-matrix moment statistics.

| Config | Layers | d_model | Heads | Context | ≈ Params |
|---|---|---|---|---|---|
| `tiny` | 4 | 128 | 4 | 256 | ~1.8 M |
| `small` | 6 | 384 | 6 | 512 | ~13.7 M |
| `base` | 8 | 512 | 8 | 512 | ~29 M |
| `large` (optional) | 12 | 768 | 12 | 512 | ~91 M |

Corpus: TinyStories, which produces clean, fast-moving loss curves at these scales.

---

## 4. Migration policy

The deliverable is not only the curve but a rule that consults it.

### 4.1 Cost model

For a candidate migration with `G` accelerators and `H` hours of training remaining:

```
saving          = (price_ondemand − price_spot) × G × H

cost(arm)       = egress_rate × bytes(arm)                        # $ per GB moved
                + (bytes(arm) / bandwidth) × G × price_ondemand/3600   # idle during drain
                + StR(arm) × step_time × G × price_ondemand/3600       # recovery waste

feasible(arm)   = bytes(arm) / bandwidth ≤ notice_seconds          # hard deadline

migrate  iff    saving > min over feasible arms of cost(arm)
chosen arm   =  argmin over feasible arms of cost(arm)
```

Two observations fall out of the structure:

- The **feasibility constraint binds before the cost constraint** at low bandwidth. An arm that is cheaper in dollars but cannot drain within the notice window is simply unavailable; §1.2 shows that at 10 Gb/s this rules out shipping the moments at 7B scale regardless of price.
- Which cost term dominates depends on the regime. Cross-region egress at ~\$0.09/GB makes the egress term dominant; same-zone transfer, where egress is often free, shifts dominance to the recovery-waste term and makes StR the deciding quantity. The policy tool reports the breakdown rather than only the decision.

All prices, bandwidths and notice windows are **parameters of a model**, not measurements. The tool takes them on the command line and states them in its output.

### 4.2 Interface

```powershell
python -m mvs.policy --curve docs/figures/pareto.json `
                     --spot-discount 0.75 --bandwidth-gbps 10 `
                     --notice-seconds 30 --egress-usd-per-gb 0.09 `
                     --devices 8 --hours-remaining 12
```

---

## 5. Results

**No experiments have been run.** These tables define the reporting format and will be filled from measured runs only.

### 5.1 Bytes vs. steps lost — `base` scale, `mid` migration point

| Arm | Protocol | Measured payload (B/param) | StR | Regret area | Recovery seconds |
|---|---|---|---|---|---|
| `full` | `cold` | TBD | TBD | TBD | TBD |
| `w_v` | `cold` | TBD | TBD | TBD | TBD |
| `w_m` | `cold` | TBD | TBD | TBD | TBD |
| `w_only` | `cold` | TBD | TBD | TBD | TBD |
| `w_only` | `frozen_warmup` | TBD | TBD | TBD | TBD |
| `w_factored_v` | `cold` | TBD | TBD | TBD | TBD |
| `w_emb_v` | `cold` | TBD | TBD | TBD | TBD |
| `cold_reset` | `cold` | TBD | TBD | TBD | TBD |

Seed band `eps` for this cell: TBD.

### 5.2 Planned figures

1. **Pareto frontier** — payload bytes (x) against steps lost (y), one point per arm × protocol, with the seed band drawn as a horizontal floor.
2. **Recovery trajectories** — loss against step around `T`, control and arms overlaid, seed band shaded.
3. **Schedule dependence** — StR against migration point, one line per arm (RQ4).
4. **Per-tensor-class breakdown** — contribution to residual loss gap by tensor class: embeddings, attention projections, MLP, LayerNorm gains (RQ3/H4).
5. **Policy regions** — bandwidth against price differential, shaded by which arm the policy selects.

---

## 6. Repository layout

```
hpcc/
├── CLAUDE.md          # working agreement, invariants, conventions — read first
├── README.md          # this file
├── configs/           # model / arm / protocol / experiment YAML
├── src/mvs/           # the package: model, data, optim, state, compress,
│                      # reconstruct, train, migrate, metrics, policy, transport
├── kernels/           # CUDA and OpenMP kernels (syllabus Units 2–3)
├── scripts/           # PowerShell entry points
├── experiments/       # run outputs (git-ignored)
├── analysis/          # plotting and table generation
├── tests/             # determinism and accounting tests
└── docs/              # report and figures
```

Only `CLAUDE.md` and `README.md` currently exist; the rest is the agreed target structure.

---

## 7. Hardware, and what it means for validity

Measured on the development machine:

- **NVIDIA GeForce RTX 4050 Laptop, 6141 MiB VRAM**, compute capability 8.9, driver 566.26
- Python 3.13.15; PyTorch currently `2.13.0+cpu` (a CUDA build must be installed — see §8)
- `torch.distributed` present with **gloo only; NCCL is unavailable on Windows**
- No CUDA Toolkit (`nvcc`) and no MPI runtime installed

Consequences, stated plainly:

- **Models run at ~2M–91M parameters, not 7B.** All payload figures are reported in **bytes per parameter**, which is scale-free; the 7B numbers in §1 are arithmetic extrapolations and are labelled as such wherever they appear.
- **Migration is simulated in-process** — serialize the chosen state, tear down model and optimizer, rebuild, resume. This reproduces the *information* lost by a real migration exactly. It does not reproduce network transfer time, which comes from the analytic cost model instead.
- **Multi-process experiments use gloo**, and exist to demonstrate the transport and collective pattern for the report, not to produce timing results.

### Threats to validity

| Threat | Mitigation |
|---|---|
| Moment statistics may not transfer from 29M to 7B parameters. | Report per-parameter and per-tensor-class quantities; run the scale axis (`tiny`→`base`) and show the *trend*, treating any extrapolation as a conjecture. |
| Short schedules may not reach the regime where `v` has settled. | Include a `late` migration point at ~85% of schedule and report `v`'s drift rate directly. |
| Single-GPU simulation omits sharded-optimizer effects (ZeRO-style state partitioning changes per-rank payloads). | Document the sharding factor explicitly in the cost model; per-rank payload is `bytes/world_size` for partitioned state. |
| Laptop thermal throttling corrupts step-time measurements feeding the cost model. | Timing from warmed-up runs only, outliers reported, no interleaved GPU work. |
| StR is sensitive to the choice of `eps` and `K`. | Report the seed band next to every StR; report regret area as a threshold-free cross-check. |

---

## 8. Getting started

```powershell
# 1. Replace the CPU-only wheel with a CUDA build (driver 566.26 supports CUDA 12.x)
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Install the package and dependencies
pip install -e .

# 3. Determinism tests must pass before any run
python -m pytest tests/ -q

# 4. Establish the noise floor FIRST
python -m mvs.train --experiment configs/experiment/seed_band.yaml

# 5. Control run, then one migration cell
python -m mvs.train --model configs/model/base.yaml --out experiments/ctrl_base_s0
python -m mvs.migrate --model configs/model/base.yaml `
                      --arm configs/arm/w_only.yaml `
                      --protocol configs/protocol/frozen_warmup.yaml `
                      --at 0.40 --control experiments/ctrl_base_s0

# 6. Build the curve and consult the policy
python -m analysis.report --runs experiments/ --out docs/figures
python -m mvs.policy --curve docs/figures/pareto.json --spot-discount 0.75 `
                     --bandwidth-gbps 10 --notice-seconds 30
```

Optional components and what they need: `kernels/cuda/` requires the CUDA Toolkit (`nvcc`); `kernels/omp/` requires a C compiler with OpenMP; `transport/mpi_shipper.py` requires MS-MPI and `mpi4py`.

---

## 9. Mapping to the HPCC syllabus

| Unit | Syllabus content | Where it appears in this project |
|---|---|---|
| **1** | Parallel architecture; shared vs. distributed memory; parallel algorithms; performance metrics | The migration problem is a distributed-memory state-consistency problem. Performance metrics are applied to migration itself: speedup and efficiency lost to recovery, and an Amdahl-style bound in which recovery steps are the serial fraction of an elastic job. |
| **2** | OpenMP essentials, data sharing and synchronization; MPI and distributed-memory computing; matrix representation and parallel matrix solvers; domain decomposition | `kernels/omp/factorize_omp.c` parallelizes the row/column factorization of `v` across the parameter matrices on the CPU. `transport/mpi_shipper.py` performs the state transfer with MPI point-to-point and collective operations. The low-rank factorization of `v` is a parallel matrix problem; data-parallel replication and optimizer-state sharding are a domain decomposition of the training state. |
| **3** | GPU architecture; GPGPU and CUDA; thread execution; matrix computing in CUDA; cuBLAS/cuDNN; accelerated case studies | `kernels/cuda/moment_recon.cu` implements the fused moment-reconstruction step; memory-bandwidth analysis of Adam's update (a bandwidth-bound elementwise kernel over `θ`, `g`, `m`, `v`) motivates the fusion. cuBLAS backs the low-rank factorization. The whole study is a case study in accelerated scientific/ML computing. |
| **4** | Cloud computing and its importance; benefits and challenges; IaaS/PaaS/SaaS; cloud architecture; cloud storage; networking and its challenges; HPC–AI synergy; training and inference on HPC | The motivation is spot-instance IaaS economics. The cost model in `policy.py` covers egress pricing, cloud storage tiers for checkpoints, and the bandwidth and notice-window constraints that make cloud networking the binding limit. The project is squarely on HPC–AI synergy: using HPC methods to make large-model training economically viable on commodity cloud capacity. |

---

## 10. Plan of work

| Phase | Output | Gate |
|---|---|---|
| 1. Scaffolding | `src/mvs/` training loop, deterministic loader, instrumented Adam | Determinism tests pass (§3.4) |
| 2. Noise floor | Seed band at each scale | `eps` reported and stable across reruns |
| 3. Endpoints | `full` and `w_only` at `base`/`mid` | A measurable gap exists above the seed band |
| 4. Full ablation | All arms × protocols at `base`; schedule axis | Pareto frontier plotted |
| 5. Scale and class | Scale sweep; per-tensor-class breakdown | Trend across scales characterized |
| 6. Policy | `policy.py` + region plot | Decision rule reproduces the arm choice from the measured curve |
| 7. Report | `docs/` report and figures | Every number satisfies the definition of done ([CLAUDE.md](CLAUDE.md) §11) |

Phase 3 is the project's go/no-go point. If the gap between `full` and `w_only` does not clear the seed band at these scales, the effect is not measurable on this hardware, and the honest outcome is to report that with the bound implied by `eps` — a clean negative result with a cost curve is a complete project.

---

## 11. Related work

This project sits between two existing lines. Elastic and preemption-tolerant training systems (Varuna, Bamboo, Oobleck, Gemini, CheckFreq) reduce the *frequency* or *latency* of checkpoint and recovery events while treating the state itself as fixed. Memory-efficient optimizers (Adafactor, SM3, 8-bit Adam) shrink optimizer state for *resident memory*, not for migration, and are evaluated on end-task quality rather than on recovery from a state discontinuity. The question here — how much state must cross a migration boundary, given the option of rebuilding the rest — is the intersection.

- Kingma & Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
- Shazeer & Stern. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* ICML 2018.
- Anil et al. *Memory Efficient Adaptive Optimization (SM3).* NeurIPS 2019.
- Dettmers et al. *8-bit Optimizers via Block-wise Quantization.* ICLR 2022.
- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020.
- Mohan et al. *CheckFreq: Frequent, Fine-Grained DNN Checkpointing.* FAST 2021.
- Athlur et al. *Varuna: Scalable, Low-cost Training of Massive Deep Learning Models.* EuroSys 2022.
- Thorpe et al. *Bamboo: Making Preemptible Instances Resilient for Affordable Training.* NSDI 2023.
- Jang et al. *Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates.* SOSP 2023.
- Wang et al. *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints.* SOSP 2023.

Citations are to be verified against the published venues before the final bibliography.
