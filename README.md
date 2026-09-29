# minstate

When you train on spot GPUs, the cloud gives you 30 seconds (GCP) to 2 minutes (AWS) of warning before it takes the machine away. If you want to keep training somewhere else, the optimizer state has to leave in that window. With Adam that's 12 bytes per parameter (weights, `m`, `v`), so a 7B model carries 80.9 GB. That doesn't fit through a 10 Gb/s link in 30 seconds.

This repo asks a simple question: how much of that state do you actually need to carry? It turns out not much. Ship the weights plus the row and column sums of `v` (34% of the bytes, or 17% with bf16 weights) and you lose about 6.6 optimizer steps. 8-bit moments lose essentially nothing. Starting Adam from scratch loses about 23 steps, and "warming up" the moments from fresh gradients is the worst option of all.

Everything here is small and readable: a nanoGPT-style model, a trainer that is deterministic per step, a payload format with CRCs, and a runtime that drains on a preemption notice and resumes somewhere else.

## install

```
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[test,vision]"
```

Dependencies are torch, numpy, pyyaml and matplotlib. Add `.[qlora]` if you want the 7B path. Run `python -m pytest` to check things work (about 90 s on CPU).

## quick start

The fastest way to see what this does is to preempt a job and watch it come back:

```
python scripts/migrate_demo.py --config configs/shakespeare_char.yaml --at 300 max_steps=800
```

This trains a small character-level GPT on tiny shakespeare twice. The first run is an uninterrupted control. The second run gets a real OS signal at step 300, drains its state to `payload.mvs` using the `w_vr1` strategy, exits, and a brand new process picks it up. At the end you get a table. This one is from the full 2000-step run with `--at 600`:

```
| event   | drained_at | payload_MB | drain_s | downtime_s | integrity |
| preempt | 602        | 43.166     | 0.247   | 4.746      | crc32 ok  |

fidelity: deviates by up to 0.0188 nats (lossy strategy)  final val gap -0.0002
```

That's a 43 MB payload instead of 129 MB, a drain of a quarter of a second, and a final loss matching the control run to within 0.0002. On an RTX 4050 laptop this takes about 5 minutes. The full 2000-step version (drop the overrides) takes about 12.

## training

```
python scripts/train.py --config configs/shakespeare_char.yaml
python scripts/train.py --config configs/shakespeare_char.yaml --resume runs/shakespeare_char/payload.mvs
```

Configs are YAML, and you can override any key with `key=value` on the command line (unknown keys are an error). Each run writes its resolved `config.yaml` and a `log.jsonl` with every step, eval, drain, checkpoint and resume, including timings.

For multi-GPU, use torchrun or the bundled launcher, which restarts the group after a crash:

```
torchrun --nproc_per_node 8 --max-restarts 3 scripts/train.py --config configs/shakespeare_char.yaml \
    --auto-resume --ckpt-every 500 --ckpt-strategy w_mq8_vlog8
python scripts/launch.py --nproc 2 --max-restarts 3 -- scripts/train.py --config configs/shakespeare_char.yaml --auto-resume
```

A preemption notice can arrive as SIGTERM, CTRL_BREAK, a sentinel file, or a thread polling EC2/GCP metadata. At the next step boundary all ranks agree on it, rank 0 writes the payload, and everyone exits with code 3 (meaning drained, not crashed). Batches, dropout masks and the learning rate are all pure functions of `(seed, step, micro-batch)`, so resuming on a different number of GPUs gives the same trajectory. Payloads are written to a temp file, fsynced and atomically renamed, and every tensor carries a CRC-32, so a torn file gets rejected instead of half-loaded.

## chaos

`chaos.py` injects a sequence of preemptions, crashes and resizes into a live job and measures what each one cost:

```
python scripts/chaos.py --config configs/shakespeare_char.yaml --world 2 \
    --events preempt@600:1,crash@900,preempt@1300:2 --strategy w_vr1 --ckpt-every 100
```

It writes `recovery.md` and `recovery.json`, with drain time, restart time, steps redone, downtime, goodput against the control run, and whether the result stayed bitwise identical.

## strategies

A strategy decides what gets shipped and how the missing parts are rebuilt at the destination. The interesting ones:

- `full`: everything in fp32. The baseline.
- `w_mq8_vlog8`: weights, plus `m` in blockwise int8 and `v` in log-domain uint8.
- `w_vr1`: weights, plus a rank-1 (row/col sums) approximation of `v`. `m` restarts at zero.
- `wbf16_vr1`: the same, with bf16 weights.
- `w_fresh`: weights only, with a fresh Adam.
- `w_warm`: weights only, with both moments estimated from fresh gradients.

The rest are listed in `mvs/state/strategy.py`. You can also build a custom strategy as `theta:m:v:rebuild`, for example `bf16:q8:rank1:none`.

## results

Here are the steps lost after migration on the largest GPT scale in the sweep. The seed-to-seed noise floor is about 1.1 steps.

| strategy | payload | steps lost |
|---|---|---|
| full | 100% | 0.0 |
| w_mq8_vlog8 | 50% | 0.4 |
| w_vr1 | 34% | 6.6 |
| wbf16_vr1 | 17% | 6.8 |
| w_fresh | 33% | 23.4 |
| w_warm | 33% | 61.2 |

All 156 migrations are in `reports/paper/summary.csv`. To rebuild the tables, or ask which strategy to use for a given model, notice window and bandwidth:

```
python scripts/analyze.py reports/paper/summary.csv
python scripts/policy.py reports/paper/summary.csv --model llama-7b --notice-s 30 --bandwidth-gbps 10 --frac 0.6
```

To rerun the sweep yourself (it's resumable, and finished cells are skipped):

```
python scripts/run_experiment.py configs/experiments/quick.yaml     # a few minutes on one GPU
python scripts/run_experiment.py configs/experiments/shakespeare_scales.yaml
```

## 7B with QLoRA

```
pip install -e ".[qlora]"
python scripts/train.py --config configs/qlora_7b.yaml
```

The base model is loaded in 4-bit NF4 (about 4 GB, so it fits on a 6 GB laptop GPU) and frozen. Only the LoRA adapters train, so only they migrate: 100.7 MB in full, 35.7 MB as `w_vr1`, 18.9 MB as `wbf16_vr1`. The payload records a fingerprint of the base model, and the destination refuses to resume if its base doesn't match. `configs/qlora_tiny.yaml` runs the same path on CPU with a random two-layer Llama.

## notes

- Windows works. There's no NCCL there, so multi-process runs use gloo. If you have fewer GPUs than ranks, `device=auto` puts every rank on CPU so the numerics stay consistent.
- Code layout: `mvs/state/` holds capture, codecs, wire format and reconstruction. `mvs/elastic.py` and `mvs/launcher.py` hold the runtime. `mvs/chaos.py` does fault injection, `mvs/policy.py` holds the cost model, and `mvs/models/` has the GPT, ResNet-18 and QLoRA wrapper.
