# mvs — minimum viable state for elastic training

A training job on spot GPUs gets 30 seconds (GCP) to 2 minutes (AWS) of warning before the machine is taken away.
With Adam, the state that must leave is 12 bytes per parameter: weights, first moment `m`, second moment `v`.
For a 7B model that is 80.9 GB, which cannot cross a 10 Gb/s link in 30 s.

This repo measures how much of that state is actually needed. The answer: the weights plus the row and column
sums of `v` (34% of the state, 17% with bf16 weights) cost about 6.6 optimiser steps. 8-bit moments cost nothing.
Rebuilding Adam from scratch costs 23 steps, and "warming up" the moments from fresh gradients is worst of all.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[test,vision]"          # add ".[qlora]" for the 7B path
python -m pytest                          # 74 tests, ~90 s on CPU
```

## One job

```bash
python scripts/train.py --config configs/shakespeare_char.yaml --strategy w_vr1
python scripts/train.py --config configs/shakespeare_char.yaml --aws-spot --strategy w_vr1
python scripts/train.py --config configs/shakespeare_char.yaml --resume runs/shakespeare_char/payload.mvs
torchrun --nproc_per_node 8 --max-restarts 3 scripts/train.py --config configs/shakespeare_char.yaml \
    --auto-resume --ckpt-every 500 --ckpt-strategy w_mq8_vlog8
python scripts/launch.py --nproc 2 --max-restarts 3 -- scripts/train.py --config configs/shakespeare_char.yaml --auto-resume
```

Config is YAML plus `key=value` overrides. Unknown keys are an error. Each run writes `config.yaml` and a JSON-lines
`log.jsonl` with every step, evaluation, drain, checkpoint and resume, including stage timings in milliseconds.

## The elastic runtime

| event | what happens |
|---|---|
| preemption notice | SIGTERM, CTRL_BREAK, a sentinel file, or a thread polling EC2 / GCP metadata sets a latch. At the next step boundary all ranks `all_reduce(MAX)` the latch, rank 0 drains the chosen strategy to `payload.mvs`, and every rank exits with code 3 (drained, not crashed). |
| hard crash | the launcher sees a non-zero exit, tears the group down, restarts it; `--auto-resume` loads the newest of `payload.mvs` and `ckpt.mvs`. |
| elastic resize | batches, dropout masks and the learning rate are pure functions of `(seed, step, micro-batch)`. A new world size changes only the accumulation factor, so the trajectory is unchanged. |

Every payload is a magic number, a JSON header and 64-byte aligned tensors with a CRC-32 each, written to a temp
file, fsynced and atomically renamed. A torn or corrupted payload is rejected, never half-loaded.

### Measuring recovery

```bash
python scripts/chaos.py --config configs/shakespeare_char.yaml --world 2 \
    --events preempt@600:1,crash@900,preempt@1300:2 --strategy w_vr1 --ckpt-every 100
python scripts/migrate_demo.py --config configs/shakespeare_char.yaml --strategy w_vr1 --at 600
```

`chaos.py` runs an uninterrupted control, then the same job while it injects the events: real signals to live
workers, `SIGKILL` for crashes, a new world size after each preemption. It writes `recovery.md` and `recovery.json`:

| column | meaning |
|---|---|
| `drained_at`, `drain_s` | step the job stopped at, and notice-to-payload-on-disk time |
| `resumed_from`, `steps_redone` | step the new job started from; work recomputed (0 for a drain, up to a checkpoint interval for a crash) |
| `restart_s`, `init_s` | process spawn to first log line; rendezvous, data and model build |
| `fetch_ms`, `rebuild_ms` | read + CRC + host-to-device + decode; reconstruction of dropped moments |
| `downtime_s` | last step before the event to first step after it |
| `fidelity` | `bitwise identical`, `identical up to summation order` (resize), or the loss deviation of a lossy strategy |
| `goodput` | useful step time over wall time, against the same number for the control |

## Strategies

| name | θ | m | v | rebuild |
|---|---|---|---|---|
| `full` | fp32 | fp32 | fp32 | – |
| `w_mq8_vlog8` | fp32 | blockwise int8 | log-domain uint8 | – |
| `w_m_vr1` | fp32 | fp32 | rank-1 | – |
| `w_m_warm` | fp32 | fp32 | – | estimate v, K=20 |
| `w_v`, `w_vr1`, `w_vsvd4`, `wbf16_vr1` | fp32 / bf16 | – | fp32 / rank-1 / rank-4 SVD | m ← 0 |
| `w_warm`, `w_warmv` | fp32 | – | – | estimate both / v only |
| `w_fresh`, `w_rewarm` | fp32 | – | – | fresh Adam, τ ← 0 (+ LR re-warm) |

Custom strategies use `theta:m:v:rebuild`, for example `bf16:q8:rank1:none`. Keeping `m` while zeroing `v` is refused.

## The study

```bash
python scripts/run_experiment.py configs/experiments/quick.yaml            # minutes on one GPU
python scripts/run_experiment.py configs/experiments/shakespeare_scales.yaml
python scripts/run_experiment.py configs/experiments/cifar_resnet18.yaml
python scripts/diagnose.py configs/experiments/shakespeare_scales.yaml
python scripts/analyze.py reports/paper/summary.csv
python scripts/policy.py reports/paper/summary.csv --model llama-7b --notice-s 30 --bandwidth-gbps 10 --frac 0.6
```

The sweep is resumable: every (scale, fork, strategy) result is written atomically and skipped if present.
`reports/paper/summary.csv` holds the 156 migrations of the report (Appendix B); `analyze.py` rebuilds Table 6.1
from it and `policy.py` rebuilds Tables 7.2 and 7.3.

## 7B with 4-bit QLoRA

```bash
pip install -e ".[qlora]"
python scripts/train.py --config configs/qlora_7b.yaml
python scripts/chaos.py --config configs/qlora_7b.yaml --events preempt@120,crash@260 --strategy w_vr1 --ckpt-every 50 max_steps=300
python scripts/policy.py reports/paper/summary.csv --model llama-7b-qlora
```

The base model is loaded in 4-bit NF4 with double quantisation (about 4 GB, so it fits a 6 GB laptop GPU) and frozen.
LoRA adapters (rank 16 on `q_proj` and `v_proj`, 8.4M parameters) are the only trainable state, so they are the
only thing that migrates: 100.7 MB in full, 35.7 MB as `w_vr1`, 18.9 MB as `wbf16_vr1`. The frozen base is not
state: the payload carries its fingerprint, and the destination re-loads it from its own cache and refuses a
mismatch. `configs/qlora_tiny.yaml` runs the same path on CPU with a random two-layer Llama.

## Layout

| path | role |
|---|---|
| `mvs/workloads.py` | shakespeare_char, HF-tokenised text, CIFAR-10; stateless SplitMix64 batches |
| `mvs/models/` | nanoGPT-style GPT, CIFAR ResNet-18, 4-bit QLoRA wrapper |
| `mvs/trainer.py` | bf16 autocast, accumulation, DDP, per-micro-batch seeding, strict determinism |
| `mvs/state/` | capture by name, codecs with exact byte counts, strategies, wire format, reconstruction |
| `mvs/elastic.py`, `mvs/launcher.py` | notices, drain, resume, periodic checkpoints, restarting launcher |
| `mvs/chaos.py` | fault injection and the recovery report |
| `mvs/experiment.py`, `mvs/analysis.py` | sweep, steps-lost estimator, noise floor, figures, first-step fidelity |
| `mvs/policy.py` | byte model for any shape list, cost model, migration decision |
