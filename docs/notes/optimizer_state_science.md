# Optimizer-state science: dropping, compressing and rebuilding Adam's moments

Scope: (1) checkpoint and optimizer-state compression, (2) memory-efficient optimizers that keep less second-moment state, (3) resets, warm-up and bias correction, (4) rebuilding the diagonal preconditioner from few gradients, (5) run-to-run noise, (6) prior bytes-versus-steps-lost work. Compiled 11 Nov 2026.

How things were checked: metadata (title, authors, venue, year, DOI, citation count) comes from the Semantic Scholar batch API (citation counts as of 11 Nov 2026). Abstracts come from the same records, and arXiv abs/HTML pages were fetched for the key papers. Numbers quoted below come from those abstracts or full-text HTML. Venue strings follow Semantic Scholar unless a publisher page was fetched. "S2 cites" means the Semantic Scholar citation count.

---

## Q1. Checkpoint compression that touches optimizer state

### Takeaway
Only a handful of checkpoint-compression papers compress Adam's m and v explicitly. ExCP (ICML 2024) is the strongest and most cited. It prunes and quantizes weights and both moments jointly, reaching about 70x on Pythia-410M. Like the rest of this line, it evaluates by resuming from compressed checkpoints and reporting final downstream accuracy. None of them measures a per-payload recovery transient (steps-to-recovery or regret) against a matched control.

### Cited Findings
- **ExCP** (Li, Chen, Shu, Tang et al., Huawei Noah's Ark; ICML 2024; arXiv 2406.11257; S2 cites 19). Pipeline: residuals between adjacent checkpoints, then weight-momentum joint pruning, then non-uniform 4-bit quantization (two int4 values packed per int8), then 7zip. It treats the Adam state explicitly as O_t = {v_t, m_t} and says it is "the first work to our knowledge that jointly considers both weights and momentum states pruning". A convergence theorem adds a regret term proportional to sum sqrt(T v_hat (1-M_o)), argued small because only entries with small v are pruned. Models: ViT-L32/ImageNet-1K, Pythia-410M (trained on about 1/20 of the 300B Pile tokens), PanGu-pi-1B and 7B. Results: Pythia-410M goes from 4.53 GB to 0.06 GB (about 70x) with downstream average 43.11 to 42.93. PanGu-pi-7B goes from 98.59 GB to 4.10 GB (about 25x) with average 56.71 to 57.09. Evaluation follows DynaQuant: "we break and resume the training process from compressed checkpoints" periodically, then report final accuracy and a training-loss-curve figure. The text found shows no seeds or error bars. — [arXiv HTML](https://arxiv.org/html/2406.11257); [ICML poster](https://icml.cc/virtual/2024/poster/33381)
- **Inshrinkerator / DynaQuant** (Agrawal, Reddy, Bhattamishra, Nookala, Vashishth, Rong, Tumanov; ACM SoCC 2024, DOI 10.1145/3698038.3698553; arXiv 2306.11800; S2 cites 13). Non-uniform quantization with a dynamic per-weight-group search, plus quantization-aware delta compression. It reports "a compression ratio up to 39x" and "withstanding up to 10 restores with negligible accuracy impact". In the HTML, the full text describes checkpoints as including "optimizer state", but its compression mechanisms are stated for model weights. I found no separate treatment of Adam moments. — [arXiv](https://arxiv.org/abs/2306.11800)
- **LC-Checkpoint** is the ICML 2020 paper "On Efficient Constructions of Checkpoints" (Yu Chen, Zhenming Liu, Bin Ren, Xin Jin; PMLR v119, pp. 1627-1636). It is a lossy delta-checkpoint scheme. Its abstract and full text were not inspected for optimizer-state handling. — [PMLR](https://proceedings.mlr.press/v119/chen20m.html)
- **Check-N-Run** (Eisenman, Matam, Ingram, Mudigere, Krishnamoorthi, Nair, Smelyanskiy, Annavaram; Meta; arXiv 2010.08679, published at NSDI 2022 per common citation, venue not verified here). Incremental checkpointing of modified embedding rows, plus quantization. It cuts "write bandwidth by 6-17x and the required capacity by 2.5-8x" on production recommendation models "without degrading training accuracy". — [arXiv](https://arxiv.org/abs/2010.08679)
- **Delta-DNN** (Hu et al., ICPP 2020, DOI 10.1145/3404397.3404408) does delta compression of floats between neighbouring training versions. The ACM page was found by search but its full author list was not fetched (partly UNVERIFIED). — [ACM](https://dl.acm.org/doi/abs/10.1145/3404397.3404408)
- **QD-Compressor** (Jin et al.). Conference version at IEEE ICCD 2021 ([IEEE 9643728](https://ieeexplore.ieee.org/document/9643728)). Journal version: "Design of a Quantization-Based DNN Delta Compression Framework for Model Snapshots and Federated Learning", IEEE TPDS 34:923 (2023) ([ADS](https://ui.adsabs.harvard.edu/abs/2023ITPDS..34..923J/abstract)). Method: layer-based local-sensitive quantization of deltas plus error feedback. The full author list was not fetched (partly UNVERIFIED).
- **Kim & Belyaev 2025** (arXiv 2506.12000; S2 cites 3). Compresses "weights and optimizer states" using the previous checkpoint as context for arithmetic coding, plus pruning and quantization, with "near-lossless training recovery from restored checkpoints". — [arXiv](https://arxiv.org/abs/2506.12000)
- **FlashOptim** (Gonzalez Ortiz, Gupta, Renard, Blalock; arXiv 2602.23349, 2026; S2 cites 6). Uses companding functions for 8-bit optimizer state and master-weight splitting. AdamW memory drops "from 16 bytes to 7 bytes per parameter, or 5 bytes with gradient release", and checkpoints shrink by "more than half", with "no measurable quality degradation", including Llama-3.1-8B fine-tuning. — [arXiv](https://arxiv.org/abs/2602.23349)
- **eXmY** (Agrawal, Hedlund, Hechtman, Google; arXiv 2405.13938). An arbitrary-bit-width float format "deployed in production for almost 2 years", applicable to "master weights and optimizer state" in checkpoints and transfers. — [arXiv](https://arxiv.org/abs/2405.13938)
- **LLMTailor** (Sun et al.; SC25 Workshops, DOI 10.1145/3731599.3767515) stores only layers with significant updates, together with their optimizer states, giving "4.3 times smaller" checkpoints for Llama3.1-8B. **LayerCheck** (same group; IEEE CLUSTER 2026, DOI 10.1109/CLUSTER68337.2026.00041) rebuilds a mixed-timestamp state with matching per-layer optimizer states. It reports a post-restart loss deviation "by at most 0.54%" from the failure-free run and "up to 22.6x" smaller checkpoints. — [arXiv 2602.22158](https://arxiv.org/abs/2602.22158); [arXiv 2609.27193](https://arxiv.org/abs/2609.27193)
- **ZipNN** (Hershcovitch et al.; IEEE CLOUD 2025; arXiv 2411.05239; S2 cites 39) is lossless model compression and does not target optimizer state. **Swift** (Zhong et al.; IEEE TPDS 2024, DOI 10.1109/TPDS.2024.3429625) recovers model plus optimizer state from data-parallel replicas or by logging and replay, rather than compressing it. — [arXiv 2411.05239](https://arxiv.org/abs/2411.05239); [arXiv 2302.06173](https://arxiv.org/abs/2302.06173)

### Inferences
- The shared evaluation norm here is the compression ratio plus the final accuracy after one or more lossy restores, without seed bands. A recovery-transient metric (StR, regret area against a bit-deterministic control) would be new to this line.
- ExCP prunes the moments where v is small. Its theorem bounds the cost by the pruned sqrt(v) mass. This is a natural baseline payload for minstate's "compressed v" arms.

### Gaps
- Full-text optimizer-state handling in LC-Checkpoint, Delta-DNN and QD-Compressor was not inspected.
- I found no 2025-2026 follow-up that directly extends ExCP to measure per-moment ablations.

---

## Q2. Memory-efficient adaptive optimizers: how much of v is redundant

### Takeaway
A consistent body of evidence says most of v is redundant, except along specific tensor dimensions. Adam-mini removes 99.9% or more of v's learning rates using Hessian-block structure. SlimAdam compresses up to 98% of second moments guided by a per-layer SNR. Adafactor, CAME and APOLLO show that rank-1 or random-projection structure suffices. The exceptions are the vocabulary (token) dimension of embeddings and the LM head, plus some LayerNorms. These carry per-row information that must not be averaged away. This matches minstate's `w_emb_v` hypothesis.

### Cited Findings
- **Adafactor** (Shazeer & Stern; ICML 2018; arXiv 1804.04235; S2 cites 1415). Keeps only per-row and per-column sums of v for matrices and "produces similar results to the baseline". It also shows that "adaptive methods can produce larger-than-desired updates when the decay rate of the second moment accumulator is too slow", and proposes update clipping and an increasing-decay schedule. — [arXiv](https://arxiv.org/abs/1804.04235)
- **SM3** (Anil, Gupta, Koren, Singer; NeurIPS 2019; arXiv 1901.11150). Sublinear-memory adaptivity with convergence guarantees. — [arXiv](https://arxiv.org/abs/1901.11150)
- **8-bit optimizers** (Dettmers, Lewis, Shleifer, Zettlemoyer; ICLR 2022; arXiv 2110.02861; S2 cites 593). Block-wise dynamic (non-linear) quantization keeps "32-bit performance" up to 1.5B-parameter language modelling. It needs a "stable embedding layer to reduce gradient variance that comes from the highly non-uniform distribution of input tokens". — [arXiv](https://arxiv.org/abs/2110.02861)
- **4-bit optimizer states** (Bingrui Li, Jianfei Chen, Jun Zhu, Tsinghua; NeurIPS 2023; arXiv 2309.01507; S2 cites 76). Moments show "complicated outlier patterns". They use smaller blocks plus row-wise and column-wise information, and identify "a zero point problem of quantizing the second moment", solved with a linear quantizer that excludes zero. — [arXiv](https://arxiv.org/abs/2309.01507)
- **CAME** (Luo, Ren, Zheng, Jiang, Jiang, You; ACL 2023; arXiv 2307.02047; S2 cites 51). A confidence-guided correction to Adafactor-style factorization that fixes its instability. Shown on BERT (batch 32,768) and GPT-2. — [arXiv](https://arxiv.org/abs/2307.02047)
- **Adam-mini** (Zhang, Chen, Li, Ding et al.; ICLR 2025; arXiv 2406.16793; S2 cites 171). "≥ 99.9% of these learning rates in v could be harmlessly removed" by partitioning parameters into Hessian-informed blocks with one learning rate each. Matches AdamW from 39M to 13B with 50% less memory. — [arXiv](https://arxiv.org/abs/2406.16793)
- **SlimAdam / "When Can You Get Away with Low Memory Adam?"** (Kalra, Kirchenbauer, Barkeshli, Goldstein; arXiv 2503.01843, 2025). Defines a layer-wise SNR_K(V) = E_K'[(E_K V)^2 / Var_K V] along fan-in or fan-out dimensions and saves "up to 98% of total second moments". Findings from the full text: embedding/LM-head second moments "show a strong aversion to compressing along the token dimension (vocabulary dimension) while favoring compression along the embedding dimension". Attention and final LayerNorms lose SNR during training, while MLP LayerNorms stay compressible. In Llama, attention and MLP RMSNorms show "consistently low SNR". — [arXiv HTML](https://arxiv.org/html/2503.01843)
- **Deconstructing What Makes a Good Optimizer for LMs** (Zhao, Morwani, Brandfonbrener, Vyas, Kakade; ICLR 2025; arXiv 2407.07972; S2 cites 51). SGD aside, Adam, Adafactor, Lion and Sophia perform comparably. The layer-wise Adalayer study concludes that "adaptivity on both the last layer and LayerNorm parameters in particular are necessary". — [arXiv](https://arxiv.org/abs/2407.07972)
- **AdaLomo** (Lv et al.; arXiv 2310.10195) finds that the adaptive learning rate "is more critical" than momentum, and factorizes v by non-negative matrix factorization. — [arXiv](https://arxiv.org/abs/2310.10195)
- **MicroAdam** (Modoranu, Safaryan, Malinovsky, Kurtic et al., IST Austria; NeurIPS 2024; arXiv 2405.15593) compresses gradients before they enter the state, using compressed error feedback. **LDAdam** (Robert, Safaryan, Modoranu, Jaggi; ICLR 2025; arXiv 2410.16103) keeps state in low-dimensional subspaces with a "projection-aware update rule for the optimizer states". — [arXiv](https://arxiv.org/abs/2405.15593); [arXiv](https://arxiv.org/abs/2410.16103)
- **GaLore** (Zhao, Zhang, Chen, Wang, Anandkumar, Tian; ICML 2024; S2 cites 607) reports "up to 65.5%" optimizer-state memory reduction. **FRUGAL** (Zmushko, Beznosikov, Takáč, Horváth; ICML 2025) runs Adam on a subspace and state-free signSGD/SGD on the rest. **APOLLO** (Zhu et al.; MLSys 2025) says even the "rank-1 variant, APOLLO-Mini" matches or beats AdamW at SGD-level memory. **Adapprox** (arXiv 2403.14958) uses randomized low-rank v with adaptive rank, saving "34.5% to 49.9%" memory on GPT-2 117M/345M. **GWT** (arXiv 2501.07237) keeps moments only for Haar approximation coefficients, cutting memory "by up to 52%". — [GaLore](https://arxiv.org/abs/2403.03507); [FRUGAL](https://arxiv.org/abs/2411.07837); [APOLLO](https://arxiv.org/abs/2412.05270); [Adapprox](https://arxiv.org/abs/2403.14958); [GWT](https://arxiv.org/abs/2501.07237)
- **No Subspace to Track** (Noel Thomas; arXiv 2607.05872, 2026). Two top-r gradient subspaces from disjoint minibatches at the same step disagree as much as ones taken T steps apart (0.73 vs 0.74 of max chordal distance, Pythia-160M, r = 128). Only about 39 of 128 directions are reproducible. "Carrying the second moment blindly is provably about (r-k*)/2 worse than the best rotation-blind estimator", while m "transports exactly through the rotation". Results use 3 seeds at 1B over 40k steps. — [arXiv](https://arxiv.org/abs/2607.05872)
- **SPAM** (Huang et al.; ICLR 2025; arXiv 2501.06842). Gradient spikes reach "up to 1000x larger than typical gradients". It resets momentum periodically and supports sparse momentum. — [arXiv](https://arxiv.org/abs/2501.06842)

### Inferences
- The project's rank-1 v result (about 6.6 steps at 34% of bytes) is consistent with Adafactor, CAME and APOLLO. SlimAdam predicts that its residual loss concentrates in the embedding and LM-head token dimension and in attention or final norms. A per-tensor-class breakdown should test exactly this.
- The 4-bit paper's zero-point problem and FlashOptim's companding are prior art for "quantize v in the log domain". Cite them when describing the 8-bit arm.

### Gaps
- None of these papers studies a *transferred* v (factorized once at migration and then evolved). They all maintain the compressed form throughout training.

---

## Q3. Resets, warm-up, bias correction and the restart transient

### Takeaway
Theory and empirics agree that the early-Adam instability comes from update magnitude when v is poorly estimated, not from "forgetting". Ma & Yarats tie warm-up to update size and recommend 2/(1-b2) steps. Kalra & Barkeshli propose initializing v as a warm-up substitute, and Abuduweili & Liu (CPAL 2025) initialize v from data. Laing & Orvieto (2025) recast bias correction as an implicit LR schedule. Resetting Adam helps when the loss changes (deep RL, Asadi et al.), but at a gap with a stationary objective the 2026 OrbitTrace study finds that dropping AdamW state costs perplexity and partial reconstruction recovers little.

### Cited Findings
- **Asadi, Fakoor, Sabach**, "Resetting the Optimizer in Deep RL" (NeurIPS 2023; arXiv 2306.17833). Moment estimates get "contaminated" when the loss changes per iteration. Resetting Adam at each iteration "significantly improves the performance" of Rainbow on Atari. — [arXiv](https://arxiv.org/abs/2306.17833)
- **Ash & Adams**, "On Warm-Starting Neural Network Training" (arXiv 1910.08475; NeurIPS 2020 per common citation, venue not verified here). Warm-starting yields "poorer generalization" than fresh initialization at similar training loss, and they propose a "shrink and perturb" style trick. — [arXiv](https://arxiv.org/abs/1910.08475)
- **Ma & Yarats** (AAAI 2021; arXiv 1910.04209; S2 cites 88) refute RAdam's variance explanation in favour of "the magnitude of the update term". They recommend "linear warmup over 2/(1-b2) training iterations". — [arXiv](https://arxiv.org/abs/1910.04209)
- **RAdam** (Liu et al.; ICLR 2020; arXiv 1908.03265; S2 cites 2349) says the adaptive LR "has problematically large variance in the early stage" and rectifies it. — [arXiv](https://arxiv.org/abs/1908.03265)
- **Kalra & Barkeshli**, "Why Warmup the Learning Rate?" (NeurIPS 2024; arXiv 2406.09405; S2 cites 94). Warm-up's benefit is tolerating a larger target LR "by forcing the network to more well-conditioned areas". They "suggest an initialization for the variance in Adam which provides benefits similar to warmup". — [arXiv](https://arxiv.org/abs/2406.09405)
- **Abuduweili & Liu**, "Revisiting the Initial Steps in Adaptive Gradient Descent Optimization" (CPAL 2025; arXiv 2412.02153). v_0 = 0 is "a significant factor" in Adam instability. Data-driven or even random non-zero v_0 stabilizes training. — [arXiv](https://arxiv.org/abs/2412.02153)
- **Laing & Orvieto**, "Adam Simplified: Bias Correction Debunked" (arXiv 2511.20516, 2025). At optimal hyperparameters bias correction gives "no improvement in final test performance". Without proper LR scheduling it "can sometimes be detrimental", and it is "a form of implicit learning rate scheduling" that depends on (b1, b2). — [arXiv](https://arxiv.org/abs/2511.20516)
- **Orvieto & Gower**, "In Search of Adam's Secret Sauce" (NeurIPS 2025; arXiv 2505.21829). Over 1500 LMs trained, b1 = b2 keeps near-optimal performance, and Adam then reads as online mean and variance estimation of the gradient. — [arXiv](https://arxiv.org/abs/2505.21829)
- **Cohen et al.**, "Adaptive Gradient Methods at the Edge of Stability" (arXiv 2207.14484). The preconditioned-Hessian sharpness equilibrates at 38/η for Adam with b1 = 0.9. Adam "can keep advancing into high-curvature regions, while adapting the preconditioner to compensate". — [arXiv](https://arxiv.org/abs/2207.14484)
- **Molybog et al.**, "A Theory on Adam Instability" (Meta; arXiv 2304.09871). Adam can enter a state with large update norm "essentially uncorrelated with the direction of descent", observed at 7B to 546B. — [arXiv](https://arxiv.org/abs/2304.09871)
- **Gupta et al.** (arXiv 2308.04014) and **Ibrahim et al.** (TMLR 2024; arXiv 2403.08763). In continual pre-training of Pythia-410M and later 10B, LR re-warming "first increases the loss" but helps downstream. Re-warming plus re-decaying plus replay matches retraining from scratch. — [arXiv](https://arxiv.org/abs/2308.04014); [arXiv](https://arxiv.org/abs/2403.08763)
- **Mitra, OrbitTrace / "When Is Availability-Aware Training Worth It?"** (arXiv 2609.22087, 2026; single author). This is the closest recent near-miss (see Q6). Full-state checkpoint plus an effective-time LR schedule "matches uninterrupted training ... exactly on a GPT-2/AdamW task". On GPT-2/WikiText-2 (5 seeds, single A100), dropping AdamW state costs perplexity, and "reconstructing a decayed first moment recovers ≈21% of the state-loss penalty ... not robustly" (a single seed gave 38.8%). In their AAT-drop arm the second moment is dropped. — [arXiv HTML](https://arxiv.org/html/2609.22087)

### Inferences (derived here, not quoted from sources)
- **Possible error in the project's transient story.** In PyTorch-style Adam the moments are updated *before* the step. So with m = v = 0 and a large preserved t, the first update after resume is already lr·(1-b1)/sqrt(1-b2)·sign(g) ≈ 3.16·lr on every coordinate. It is not ≈0. CLAUDE.md §7 says the first update is ≈0 and the spike arrives on the second step. Check this against `optim.py`.
- **Keeping t with zeroed moments amounts to removing bias correction.** If t were reset, bias correction would give exactly lr·sign(g) on the first step and properly normalized steps after. Keeping t leaves v under-filled by a factor (1-b2^k) after k steps, inflating step size by about 1/sqrt(1-b2^k): roughly 10x at k = 10 and 3.2x at k = 100, fading over about 1/(1-b2) = 1000 steps. That is a long LR spike, consistent with Laing & Orvieto's "bias correction = implicit schedule" view. The project's §6.4 invariant ("t always carried") may therefore *cause* part of the fresh-Adam penalty. A `cold_reset` arm (t reset) or a local bias-correction counter for rebuilt moments is needed to attribute it.
- **Why "re-estimate from 20 fresh gradients" can be worst.** After 20 frozen accumulation steps with t preserved, m is about 88% formed (1-0.9^20) while v is only about 2% filled (1-0.999^20 ≈ 0.0198). The first real update is then about 1/sqrt(0.0198) ≈ 7x the steady-state step, aligned with a now-coherent mean gradient. A fresh Adam's m starts near zero instead. The fix to test is dividing the re-estimated v by (1-b2^20), or setting v to the plain mean of g^2 over the 20 samples.

### Gaps
- No paper found isolates "zero moments, keep t" against "zero moments, reset t" at a mid-training resume.

---

## Q4. Estimating the diagonal preconditioner from few gradients

### Takeaway
No paper found gives a sample-complexity curve for rebuilding Adam's v to a given accuracy mid-training. The relevant anchors are Sophia (a diagonal Hessian refreshed every handful of steps), the empirical-Fisher critique, the GaLore non-identifiability result and data-driven v_0 initialization.

### Cited Findings
- **Kunstner, Balles, Hennig** (NeurIPS 2019; arXiv 1905.12558; S2 cites 337): the empirical Fisher "does not generally capture second-order information". — [arXiv](https://arxiv.org/abs/1905.12558)
- **Sophia** (Liu, Li, Hall, Liang, Ma; ICLR 2024; arXiv 2305.14342) estimates the diagonal Hessian "every handful of iterations", EMA-smooths it and clips elementwise. It reports a 2x step speed-up over Adam on GPT 125M to 1.5B. — [arXiv](https://arxiv.org/abs/2305.14342)
- **AdaHessian** (Yao et al.; AAAI 2021; arXiv 2006.00719) uses a Hutchinson diagonal-Hessian estimator. **K-FAC** (Martens & Grosse; ICML 2015; arXiv 1503.05671) is the classic factored-curvature anchor. **FAdam** (Hwang; arXiv 2405.12807) frames Adam as natural gradient with a diagonal empirical Fisher. — [AdaHessian](https://arxiv.org/abs/2006.00719); [K-FAC](https://arxiv.org/abs/1503.05671); [FAdam](https://arxiv.org/abs/2405.12807)
- **No Subspace to Track** (arXiv 2607.05872): under N-fold averaging the gradient's spectral tail shrinks as N^(-1/4) rather than N^(-1/2), so "no averaging budget makes the subspace well defined". — [arXiv](https://arxiv.org/abs/2607.05872)

### Inferences (derived here)
- The steady-state EMA with b2 = 0.999 has an effective sample size of (1+b2)/(1-b2) ≈ 2000 gradients. Twenty samples are about 1% of that. For zero-mean Gaussian gradients, the per-coordinate relative standard deviation of a mean-of-g^2 estimate is sqrt(2/n), about 0.32 at n = 20, and worse under the heavy tails SPAM reports. Because the update divides by sqrt(v), coordinates where v is underestimated get disproportionately large steps.

### Gaps
- No empirical "rebuild error of v versus number of gradients" study exists for transformers. minstate could contribute one cheaply.

---

## Q5. Run-to-run noise for a noise floor

### Takeaway
Seed and nondeterminism variance is large and well documented. PolyPythias offers 50 public LM runs (14M to 410M) for calibrating a seed band. Summers & Dinneen show that even a one-bit change in initial weights diverges.

### Cited Findings
- **Summers & Dinneen** (ICML 2021; arXiv 2103.04514): "all sources of nondeterminism have similar effects", and "even one-bit changes in initial parameters result in models converging to vastly different values". — [arXiv](https://arxiv.org/abs/2103.04514)
- **Bouthillier et al.** (MLSys 2021; arXiv 2103.03098; S2 cites 379): accounting for variance sources in benchmarks. **Picard** (arXiv 2109.08203): seed effects in vision. — [arXiv](https://arxiv.org/abs/2103.03098); [arXiv](https://arxiv.org/abs/2109.08203)
- **PolyPythias** (van der Wal, Lesci, Müller-Eberstein, Saphra et al.; ICLR 2025; arXiv 2503.09543): 45 new runs, "9 new seeds across 5 model sizes, from 14M to 410M", plus 5 existing ones, about 7k checkpoints. They find "highly consistent training dynamics" but identifiable outlier runs. — [arXiv](https://arxiv.org/abs/2503.09543)
- OrbitTrace (arXiv 2609.22087) warns that a single seed gave 38.8% recovery against a 5-seed mean of about 21%. — [arXiv HTML](https://arxiv.org/html/2609.22087)

### Inferences
- The OrbitTrace single-seed versus five-seed gap backs minstate's own rule that a single-seed result is not a result. The preliminary 0.4 / 6.6 / 23 / 61-step numbers need the seed band before any claim.

---

## Q6. Has anyone measured bytes versus steps lost for partial optimizer state?

### Takeaway
I found no paper that measures a bytes-transferred against steps-to-recovery (or regret) curve across partial optimizer-state payloads at a migration or resume point. The closest near-misses are listed below with exactly what each did and did not measure.

### Cited Findings (near-misses)
- **OrbitTrace (arXiv 2609.22087, 2026).** What it did: interruption gaps on CIFAR-10/ResNet-18 (3 seeds) and GPT-2/WikiText-2 with AdamW (5 seeds). It compared full-state checkpointing, a weights-only checkpoint ("+drop"), Elastic (zero momentum plus LR warm-up) and AAT-drop (drop the second moment, keep a decayed first moment, warm up), measuring final accuracy or perplexity. On CIFAR, Checkpoint-Strong scored 86.07 ± 0.31, +drop 85.50 ± 0.28, Elastic 85.12 ± 0.11 and AAT-drop 85.63 ± 0.48. What it did not do: no byte accounting, no recovery-transient metric (only end-of-run quality), no quantized or factorized moments, no w_v versus w_m contrast, no moment re-estimation by frozen gradient passes, and no bit-deterministic control. Its data iterator re-creation "adds order noise". — [arXiv HTML](https://arxiv.org/html/2609.22087)
- **ExCP / Inshrinkerator.** Byte-versus-final-accuracy trade-offs for lossy checkpoints, with restores. No steps-to-recovery, no arm-by-arm moment ablation. — [ExCP](https://arxiv.org/html/2406.11257); [Inshrinkerator](https://arxiv.org/abs/2306.11800)
- **LayerCheck.** Measures post-restart loss deviation (≤0.54%) with stale per-layer state. No partial moments. — [arXiv](https://arxiv.org/abs/2609.27193)
- **Asadi et al. 2023.** Full reset versus keep, in RL. No bytes, no partial state. — [arXiv](https://arxiv.org/abs/2306.17833)
- **No Subspace to Track.** Compares transporting m versus v across subspace refreshes, but within training rather than at migration, and without bytes. — [arXiv](https://arxiv.org/abs/2607.05872)

### Inferences
- The novelty claim "first bytes-versus-recovery Pareto curve over optimizer-state payloads, with a seed-band noise floor and a bit-deterministic control" survives this search. OrbitTrace must be cited and contrasted, since it is the closest and very recent (2026).

### Gaps
- arXiv API and OpenAlex searches were rate-limited partway through, so systems-venue papers from 2025-2026 (OSDI, NSDI, EuroSys, SoCC) may be under-covered. Elastic and spot-training systems papers are another researcher's scope.

---

## Evaluation norms in the key papers

| Paper | Metrics | Baselines | Scale / data | Statistics |
|---|---|---|---|---|
| ExCP | compression ratio; downstream accuracy after resume; loss-curve figure | CNR+, QD+, DynaQuant, Residual+7Zip | ViT-L32/ImageNet; Pythia-410M (≈15B Pile tokens); PanGu-pi 1B/7B | no seeds or error bars found in the text |
| Inshrinkerator | compression ratio; accuracy after up to 10 restores | prior lossy and delta compressors | per arXiv abstract; details not extracted | not extracted |
| SlimAdam | SNR_K trajectories; final loss; % of v saved (up to 98%) | Adam, Adafactor-like | GPT-small/OpenWebText, Llama, ResNet | full text not mined for seeds |
| Adam-mini | loss; throughput (49.6% higher on Llama-2-7B) | AdamW | 39M to 13B | per abstract |
| OrbitTrace | final accuracy / perplexity; % of penalty recovered | Continuous, Checkpoint(-Strong), Elastic | ResNet-18/CIFAR-10; GPT-2/WikiText-2 | mean ± std, 3 and 5 seeds |
| No Subspace to Track | chordal distance; perplexity | GaLore variants, LDAdam, full-rank | 70M to 6.9B; 1B for 40k steps | 3 seeds |
| Kalra & Barkeshli 2024 | loss/accuracy against target LR; sharpness | warm-up variants | SGD and Adam sweeps | per abstract |
| Ma & Yarats | final performance | RAdam, warm-up schedules | image, LM, MT | per abstract |

---

## (a) Ranked shortlist to build on

1. **OrbitTrace (2609.22087).** The closest prior result. Use it as the direct contrast: it found that state loss costs and partial reconstruction recovers about 21%, but measured no bytes and no transient.
2. **ExCP (2406.11257, ICML 2024).** The canonical "compress m and v in checkpoints" paper and the strongest compression baseline. Its pruning rule (drop small-v entries) is an arm candidate.
3. **SlimAdam (2503.01843).** Its SNR analysis predicts which tensors need a full v (the embedding token dimension, attention and final norms). Use it to design and explain per-tensor-class results and the `w_emb_v` arm.
4. **Adafactor (1804.04235) together with CAME (2307.02047).** The source of the rank-1 v estimator and of the "slow b2 gives oversized updates" observation behind the transient.
5. **Laing & Orvieto 2025 (2511.20516), Ma & Yarats (1910.04209), Kalra & Barkeshli (2406.09405) and Abuduweili & Liu (2412.02153).** The theory set for the restart spike and for v-initialization fixes. These bear directly on whether t should be carried.
6. **8-bit (2110.02861) and 4-bit (2309.01507) optimizers.** Prior art for the quantized-moment arm, including the zero-point and log-domain handling of v.

## (b) Gaps where nothing was found
- No bytes-versus-steps-to-recovery or regret curve for partial optimizer state at migration or resume.
- No controlled study of "zero moments, keep t" against "zero moments, reset t" mid-training.
- No sample-complexity study of rebuilding v from n fresh gradients in transformers.
- No study of a v factorized *once* at a boundary and then evolved at full rank.
- No paper uses a bit-deterministic control plus a seed band to define recovery after a resume.

## (c) Web access
Web access worked partly. The Semantic Scholar batch API worked twice (metadata and citation counts); its search endpoint stalled. The arXiv export API worked for two searches and then returned HTTP 429. OpenAlex returned 429 on every query, and the DBLP search API returned non-JSON or HTTP 500. arxiv.org abs/HTML pages and PMLR fetched fine, and DuckDuckGo via `ddgs` worked intermittently. Partly unverified items: the Delta-DNN and QD-Compressor author lists; the Ash & Adams NeurIPS 2020 and Check-N-Run NSDI 2022 venues; full texts of LC-Checkpoint, Delta-DNN and QD-Compressor.

---

## BibTeX (verified entries)

```bibtex
@inproceedings{li2024excp, title={ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking}, author={Li, Wenshuo and Chen, Xinghao and Shu, Han and Tang, Yehui and Wang, Yunhe}, booktitle={ICML}, year={2024}, eprint={2406.11257}}
@inproceedings{agrawal2024inshrinkerator, title={Inshrinkerator: Compressing Deep Learning Training Checkpoints via Dynamic Quantization}, author={Agrawal, Amey and Reddy, Sameer and Bhattamishra, Satwik and Nookala, Venkata Prabhakara Sarath and Vashishth, Vidushi and Rong, Kexin and Tumanov, Alexey}, booktitle={ACM SoCC}, year={2024}, doi={10.1145/3698038.3698553}, eprint={2306.11800}}
@inproceedings{chen2020lccheckpoint, title={On Efficient Constructions of Checkpoints}, author={Chen, Yu and Liu, Zhenming and Ren, Bin and Jin, Xin}, booktitle={ICML}, series={PMLR}, volume={119}, pages={1627--1636}, year={2020}}
@article{eisenman2020checknrun, title={Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models}, author={Eisenman, Assaf and Matam, Kiran Kumar and Ingram, Steven and Mudigere, Dheevatsa and Krishnamoorthi, Raghuraman and Nair, Krishnakumar and Smelyanskiy, Misha and Annavaram, Murali}, journal={arXiv:2010.08679}, year={2020}}
@inproceedings{hu2020deltadnn, title={Delta-DNN: Efficiently Compressing Deep Neural Networks via Exploiting Floats Similarity}, author={Hu, Zhenbo and others}, booktitle={ICPP}, year={2020}, doi={10.1145/3404397.3404408}, note={author list not verified}}
@article{jin2023qdcompressor, title={Design of a Quantization-Based DNN Delta Compression Framework for Model Snapshots and Federated Learning}, author={Jin, Haoyu and others}, journal={IEEE TPDS}, volume={34}, pages={923}, year={2023}, note={author list not verified}}
@article{kim2025ckptcompress, title={An Efficient Compression of Deep Neural Network Checkpoints Based on Prediction and Context Modeling}, author={Kim, Yuriy and Belyaev, Eugeniy}, journal={arXiv:2506.12000}, year={2025}}
@article{gonzalez2026flashoptim, title={FlashOptim: Optimizers for Memory Efficient Training}, author={Gonzalez Ortiz, Jose Javier and Gupta, Abhay and Renard, Chris and Blalock, Davis}, journal={arXiv:2602.23349}, year={2026}}
@article{agrawal2024exmy, title={eXmY: A Data Type and Technique for Arbitrary Bit Precision Quantization}, author={Agrawal, Aditya and Hedlund, Matthew and Hechtman, Blake}, journal={arXiv:2405.13938}, year={2024}}
@inproceedings{sun2025llmtailor, title={LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models}, author={Sun, Minqiu and Huang, Xin and Guo, Luanzheng and Tallent, Nathan R. and others}, booktitle={SC25 Workshops}, year={2025}, doi={10.1145/3731599.3767515}, eprint={2602.22158}}
@inproceedings{sun2026layercheck, title={LayerCheck: Adaptive Layer-Wise Checkpointing for Large Language Model Post-Training}, author={Sun, Minqiu and Huang, Xin and Guo, Luanzheng and Tallent, Nathan R. and others}, booktitle={IEEE CLUSTER}, year={2026}, doi={10.1109/CLUSTER68337.2026.00041}, eprint={2609.27193}}
@inproceedings{hershcovitch2025zipnn, title={ZipNN: Lossless Compression for AI Models}, author={Hershcovitch, Moshik and Wood, Andrew and Choshen, Leshem and Girmonsky, Guy and others}, booktitle={IEEE CLOUD}, year={2025}, doi={10.1109/CLOUD67622.2025.00028}, eprint={2411.05239}}
@article{zhong2024swift, title={Swift: Expedited Failure Recovery for Large-Scale DNN Training}, author={Zhong, Yuchen and Sheng, Guangming and Liu, Juncheng and Yuan, Jinhui and Wu, Chuan}, journal={IEEE TPDS}, year={2024}, doi={10.1109/TPDS.2024.3429625}, eprint={2302.06173}}
@inproceedings{shazeer2018adafactor, title={Adafactor: Adaptive Learning Rates with Sublinear Memory Cost}, author={Shazeer, Noam and Stern, Mitchell}, booktitle={ICML}, year={2018}, eprint={1804.04235}}
@inproceedings{anil2019sm3, title={Memory-Efficient Adaptive Optimization}, author={Anil, Rohan and Gupta, Vineet and Koren, Tomer and Singer, Yoram}, booktitle={NeurIPS}, year={2019}, eprint={1901.11150}}
@inproceedings{dettmers2022eightbit, title={8-bit Optimizers via Block-wise Quantization}, author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke}, booktitle={ICLR}, year={2022}, eprint={2110.02861}}
@inproceedings{li2023fourbit, title={Memory Efficient Optimizers with 4-bit States}, author={Li, Bingrui and Chen, Jianfei and Zhu, Jun}, booktitle={NeurIPS}, year={2023}, eprint={2309.01507}}
@inproceedings{luo2023came, title={CAME: Confidence-guided Adaptive Memory Efficient Optimization}, author={Luo, Yang and Ren, Xiaozhe and Zheng, Zangwei and Jiang, Zhuo and Jiang, Xin and You, Yang}, booktitle={ACL}, year={2023}, eprint={2307.02047}}
@inproceedings{zhang2025adammini, title={Adam-mini: Use Fewer Learning Rates To Gain More}, author={Zhang, Yushun and Chen, Congliang and Li, Ziniu and Ding, Tian and others}, booktitle={ICLR}, year={2025}, eprint={2406.16793}}
@inproceedings{modoranu2024microadam, title={MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence}, author={Modoranu, Ionut-Vlad and Safaryan, Mher and Malinovsky, Grigory and Kurtic, Eldar and others}, booktitle={NeurIPS}, year={2024}, eprint={2405.15593}}
@inproceedings{robert2025ldadam, title={LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics}, author={Robert, Thomas and Safaryan, Mher and Modoranu, Ionut-Vlad and Jaggi, Martin}, booktitle={ICLR}, year={2025}, eprint={2410.16103}}
@article{lv2023adalomo, title={AdaLomo: Low-memory Optimization with Adaptive Learning Rate}, author={Lv, Kai and Yan, Hang and Guo, Qipeng and Lv, Haijun and Qiu, Xipeng}, journal={arXiv:2310.10195}, year={2023}}
@inproceedings{zmushko2025frugal, title={FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training}, author={Zmushko, Philip and Beznosikov, Aleksandr and Tak{\'a}{\v{c}}, Martin and Horv{\'a}th, Samuel}, booktitle={ICML}, year={2025}, eprint={2411.07837}}
@article{kalra2025slimadam, title={When Can You Get Away with Low Memory Adam?}, author={Kalra, Dayal Singh and Kirchenbauer, John and Barkeshli, Maissam and Goldstein, Tom}, journal={arXiv:2503.01843}, year={2025}}
@inproceedings{zhao2024galore, title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection}, author={Zhao, Jiawei and Zhang, Zhenyu and Chen, Beidi and Wang, Zhangyang and Anandkumar, Anima and Tian, Yuandong}, booktitle={ICML}, year={2024}, eprint={2403.03507}}
@inproceedings{zhu2025apollo, title={APOLLO: SGD-like Memory, AdamW-level Performance}, author={Zhu, Hanqing and Zhang, Zhenyu and Cong, Wenyan and Liu, Xi and others}, booktitle={MLSys}, year={2025}, eprint={2412.05270}}
@inproceedings{zhao2025deconstructing, title={Deconstructing What Makes a Good Optimizer for Language Models}, author={Zhao, Rosie and Morwani, Depen and Brandfonbrener, David and Vyas, Nikhil and Kakade, Sham}, booktitle={ICLR}, year={2025}, eprint={2407.07972}}
@article{zhao2024adapprox, title={Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices}, author={Zhao, Pengxiang and Li, Ping and Gu, Yingjie and Zheng, Yi and others}, journal={arXiv:2403.14958}, year={2024}}
@article{wen2025gwt, title={GWT: Scalable Optimizer State Compression for Large Language Model Training}, author={Wen, Ziqing and Luo, Ping and Wang, Jiahuan and Zeng, Junling and others}, journal={arXiv:2501.07237}, year={2025}}
@article{thomas2026nosubspace, title={No Subspace to Track: Non-Identifiability and Optimizer State in Low-Rank Training}, author={Thomas, Noel}, journal={arXiv:2607.05872}, year={2026}}
@article{vyas2024soap, title={SOAP: Improving and Stabilizing Shampoo using Adam}, author={Vyas, Nikhil and Morwani, Depen and Zhao, Rosie and Shapira, Itai and others}, journal={arXiv:2409.11321}, year={2024}}
@inproceedings{huang2025spam, title={SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training}, author={Huang, Tianjin and Zhu, Ziquan and Jin, Gaojie and Liu, Lu and others}, booktitle={ICLR}, year={2025}, eprint={2501.06842}}
@inproceedings{asadi2023resetting, title={Resetting the Optimizer in Deep RL: An Empirical Study}, author={Asadi, Kavosh and Fakoor, Rasool and Sabach, Shoham}, booktitle={NeurIPS}, year={2023}, eprint={2306.17833}}
@article{ash2019warmstart, title={On Warm-Starting Neural Network Training}, author={Ash, Jordan T. and Adams, Ryan P.}, journal={arXiv:1910.08475}, year={2019}, note={NeurIPS 2020 venue not verified here}}
@inproceedings{ma2021untunedwarmup, title={On the Adequacy of Untuned Warmup for Adaptive Optimization}, author={Ma, Jerry and Yarats, Denis}, booktitle={AAAI}, year={2021}, eprint={1910.04209}}
@inproceedings{liu2020radam, title={On the Variance of the Adaptive Learning Rate and Beyond}, author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei}, booktitle={ICLR}, year={2020}, eprint={1908.03265}}
@inproceedings{kalra2024warmup, title={Why Warmup the Learning Rate? Underlying Mechanisms and Improvements}, author={Kalra, Dayal Singh and Barkeshli, Maissam}, booktitle={NeurIPS}, year={2024}, eprint={2406.09405}}
@inproceedings{abuduweili2025adaminit, title={Revisiting the Initial Steps in Adaptive Gradient Descent Optimization}, author={Abuduweili, Abulikemu and Liu, Changliu}, booktitle={CPAL}, year={2025}, eprint={2412.02153}}
@article{laing2025biascorrection, title={Adam Simplified: Bias Correction Debunked}, author={Laing, Sam and Orvieto, Antonio}, journal={arXiv:2511.20516}, year={2025}}
@article{stjohn2021adamd, title={AdamD: Improved bias-correction in Adam}, author={St. John, John}, journal={arXiv:2110.10828}, year={2021}}
@inproceedings{orvieto2025secretsauce, title={In Search of Adam's Secret Sauce}, author={Orvieto, Antonio and Gower, Robert}, booktitle={NeurIPS}, year={2025}, eprint={2505.21829}}
@article{cohen2022aeos, title={Adaptive Gradient Methods at the Edge of Stability}, author={Cohen, Jeremy M. and Ghorbani, Behrooz and Krishnan, Shankar and Agarwal, Naman and others}, journal={arXiv:2207.14484}, year={2022}}
@article{molybog2023adaminstability, title={A Theory on Adam Instability in Large-Scale Machine Learning}, author={Molybog, Igor and Albert, Peter and Chen, Moya and DeVito, Zachary and others}, journal={arXiv:2304.09871}, year={2023}}
@article{gupta2023rewarm, title={Continual Pre-Training of Large Language Models: How to (re)warm your model?}, author={Gupta, Kshitij and Th{\'e}rien, Benjamin and Ibrahim, Adam and Richter, Mats L. and others}, journal={arXiv:2308.04014}, year={2023}}
@article{ibrahim2024continual, title={Simple and Scalable Strategies to Continually Pre-train Large Language Models}, author={Ibrahim, Adam and Th{\'e}rien, Benjamin and Gupta, Kshitij and Richter, Mats L. and others}, journal={TMLR}, year={2024}, eprint={2403.08763}}
@article{mitra2026orbittrace, title={When Is Availability-Aware Training Worth It? A Benchmark and Empirical Study of Interruption-Resilient Optimization Under Predictable Compute Schedules}, author={Mitra, Subhadip}, journal={arXiv:2609.22087}, year={2026}}
@inproceedings{kunstner2019empfisher, title={Limitations of the Empirical Fisher Approximation for Natural Gradient Descent}, author={Kunstner, Frederik and Balles, Lukas and Hennig, Philipp}, booktitle={NeurIPS}, year={2019}, eprint={1905.12558}}
@inproceedings{liu2024sophia, title={Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training}, author={Liu, Hong and Li, Zhiyuan and Hall, David and Liang, Percy and Ma, Tengyu}, booktitle={ICLR}, year={2024}, eprint={2305.14342}}
@inproceedings{yao2021adahessian, title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning}, author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Mustafa, Mustafa and others}, booktitle={AAAI}, year={2021}, eprint={2006.00719}}
@inproceedings{martens2015kfac, title={Optimizing Neural Networks with Kronecker-factored Approximate Curvature}, author={Martens, James and Grosse, Roger}, booktitle={ICML}, year={2015}, eprint={1503.05671}}
@article{hwang2024fadam, title={FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information}, author={Hwang, Dongseong}, journal={arXiv:2405.12807}, year={2024}}
@inproceedings{summers2021nondeterminism, title={Nondeterminism and Instability in Neural Network Optimization}, author={Summers, Cecilia and Dinneen, Michael J.}, booktitle={ICML}, year={2021}, eprint={2103.04514}}
@inproceedings{bouthillier2021variance, title={Accounting for Variance in Machine Learning Benchmarks}, author={Bouthillier, Xavier and Delaunay, Pierre and Bronzi, Mirko and Trofimov, Assya and others}, booktitle={MLSys}, year={2021}, eprint={2103.03098}}
@article{picard2021seed, title={Torch.manual\_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision}, author={Picard, David}, journal={arXiv:2109.08203}, year={2021}}
@inproceedings{vanderwal2025polypythias, title={PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs}, author={van der Wal, Oskar and Lesci, Pietro and M{\"u}ller-Eberstein, Max and Saphra, Naomi and others}, booktitle={ICLR}, year={2025}, eprint={2503.09543}}
```

Note on BibTeX: author lists ending in "others" were truncated by the metadata source. Expand them from the arXiv page before submission. Given names marked as known (for example Yunhe Wang on ExCP, Chuan Wu on Swift, Sham Kakade on Deconstructing, Tengyu Ma on Sophia) were not all shown in the truncated S2 listing. Check them against the arXiv page before use.
