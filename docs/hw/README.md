# Homework: the ten papers this work stands on

The PDFs in this folder are the ten papers the manuscript leans on hardest. Read them in this order: the first three
are the optimizer, the next four are the systems that move its state, the last three are the closest neighbours.

The summaries below follow how each paper is used and cited in our manuscript. Check a claim against the PDF before
quoting it anywhere else.

## What they did, and what we add

| # | Paper | What they did | What we add on top |
|---|---|---|---|
| 01 | Adam (Kingma and Ba, 2015) | Defined the optimizer: a running average of the gradient (`m`), one of its square (`v`), and a bias correction driven by a step counter. | We ask which of those three things a run needs after an interruption, and work out the size of the first updates when each is missing. |
| 02 | Adafactor (Shazeer and Stern, 2018) | Trains from scratch keeping only the row and column sums of `v`, for the whole run. | We apply the same estimator once, at the migration boundary, to a run that was trained with exact Adam. The sketch cannot be told from the exact `v` (0.31 against 0.29 of the seed band). |
| 03 | 8-bit Optimizers (Dettmers et al., 2022) | Stores both moments in eight bits throughout training with block-wise quantisation. | We show the same holds for a one-off hand-over: eight-bit moments at half the bytes stay at 0.09 of the seed band, with every run inside it. |
| 04 | ExCP (Li et al., 2024) | Compresses checkpoints by pruning weights and moments jointly, and reports final accuracy after restoring. | They shrink what is stored and judge it at the end of training. We ask what has to exist at all and measure the loss at every step after the restore. |
| 05 | Can't Be Late (Wu et al., NSDI 2024) | Decides when to use spot or on-demand capacity under a deadline. Everything that happens at a switch is one constant, the changeover delay. | We open that constant up: it becomes a list of (bytes, recovery cost) pairs, one per strategy, and the scheduler picks one that fits the notice. |
| 06 | Parcae (Duan et al., NSDI 2024) | Keeps training alive on spot instances by re-planning ahead of preemptions; holds a CPU-side copy of parameters and optimizer state. | Parcae moves the full state. We measure what happens when less is moved. |
| 07 | EasyScale (Li et al., SC 2023) | Elastic resizing that reproduces the fixed-size run bit for bit. | We borrow that bitwise standard and use it as an instrument: because the full-state resume is exact, any gap in another row was caused by the missing state. |
| 08 | ETC (Liu et al., 2026) | Migrates state directly between devices with no checkpoint, minimising data movement during reconfiguration. States that parameters and optimizer state must be migrated. | We test that statement. The first moment need not be migrated, and the second can be sent as a sketch. Their work makes moving bytes faster; ours makes the bytes fewer. The two compose. |
| 09 | Guarded compression (Zhang et al., 2026) | Compresses both moments under a bound on the error of the first restored update; cuts moment payloads by about 70%. | We use their first-update test as a diagnostic and find it safe but incomplete: dropping `m` fails the test and is harmless, and two strategies with the same first update differ by a factor of 2.6 afterwards. They never drop or rebuild a moment. |
| 10 | Interruption study (Mitra, 2026) | Trains under scheduled interruptions; finds that a weights-only checkpoint costs final quality and that rebuilding `m` recovers little, not robustly. | We explain both findings. "Weights only" is five different outcomes spanning a factor of thirty depending on what the destination does, and `m` cannot be rebuilt at a standstill because it records the path. We also separate `m` from `v`, count bytes, and keep data order fixed. |

## Our contributions, strongest first

1. **The two moments are not interchangeable, and the cheap one to send is the one that matters.** Keeping `v` and
   dropping `m` peaks at 0.29 of the seed band. Keeping `m` and dropping `v`, at the same bytes, peaks at 69 bands.
   With `v` sent as row and column sums the payload is a third of the full state, or a sixth with half-precision
   weights, and no run leaves the noise floor. This is the headline: nobody had separated the moments at a migration
   boundary before.
2. **A closed-form account of the arrival transient that matches measurement with no fitted parameter.** The first
   update after a cold arrival is about 4.4 times too large, not near zero as is often assumed. Keeping the step
   counter makes it decay on the long horizon of `v`; resetting it makes it decay on the short horizon of `m`
   (11.4 against 4.4 bands). Measured update sizes follow the formulas step by step.
3. **A first moment cannot be rebuilt at the destination.** Re-estimating `m` from gradients at a fixed point gives
   a vector 4.0 times too large, because the cancellation that the running average performs along a trajectory does
   not happen at a standstill. Rebuilding only `v` works (0.40 of the band); rebuilding both is worse than doing
   nothing (8.4 bands). This turns a puzzling negative result in the literature into a mechanism.
4. **The measurement method.** A bit-identical control, a seed band as the noise floor, and a censored recovery time.
   It is cheap, it is reusable as an acceptance test for any lossy checkpoint, and it is why the numbers above can be
   trusted.
5. **A migration policy.** A cost model that takes the measured curve, a notice window, a bandwidth and a price and
   returns what to send. Its outputs for a 7B model are extrapolations and are labelled so.
6. **Supporting measurements.** Bytes taken from the serialiser (within 0.04% of the analytic count), per-stage drain
   and restore latency, resume across a CPU and two GPUs, and an end-to-end run with injected preemptions and a crash.

## Where it stands with reviewers

Contributions 1 to 3 are the ones a reviewer will remember, because each is a clean, falsifiable statement with a
mechanism behind it. Contribution 4 is what makes them believable.

Four follow-up experiments, all finished on 2026-11-11, back these up:

- **beta2 sweep** (108 migrations). With the counter kept, the damage of dropping both moments grows with beta2
  (2.0, 13.1, 48.3 bands at 0.95, 0.99, 0.999); with it reset it stays flat (5.3, 6.4, 5.2). The formula predicted
  both, including the crossover at 0.95.
- **An 85M model** (30 migrations, three seeds). Same ordering as at 25M: eight-bit moments 0.17 of the band, rank-one
  sketch 0.45, both moments dropped 17.3 bands with the counter kept and 8.8 with it reset.
- **A paced link between two GPUs** (96 transfers, NVIDIA to Intel Arc, two processes). At 100 Mb/s the 85M full state
  needs 83 s and misses a 30 s notice; the rank-one payload leaves in 27.8 s and the half-precision one in 14.1 s.
  Measured drain is within 3.7% of bytes over bandwidth. The link is emulated; the devices are real.
- **A second machine.** The laptop's payloads resumed on a Kaggle T4 stay within half the seed band.

What reviewers will still push on is scale: one laptop for all training, 85M parameters at most, three seeds, and no
timed migration between two machines over a real network. The paper says all of this itself.
