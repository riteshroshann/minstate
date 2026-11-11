# Cross-device portability of deep-learning training state: what can honestly be claimed and measured

Research note for minstate. All pages were fetched on 2026-11-11 (IST; the arXiv API reported 2026-11-10T18:54Z) unless marked otherwise. "Verified" means the primary page (arXiv API record, Semantic Scholar record, or the official doc page) was fetched and read. Anything not confirmed that way is marked UNVERIFIED.

## 1. Which accelerator classes support TRAINING (backward + optimizer step), and through which backend?

### Takeaway
CPU, NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS and Google TPU all support full training in PyTorch and/or JAX. DirectML says it does, but only as a public preview. NPUs are a different story: the mainstream vendor stacks (Intel OpenVINO, Qualcomm QNN, AMD Ryzen AI, Apple Core ML) document them as inference targets. The only NPU "training" in the literature is research that either goes around the vendor stack (Apple ANE through private APIs) or avoids backprop altogether (zeroth-order or forward-gradient training on fixed-point NPUs). **On this laptop's Intel AI Boost NPU, there is no supported way to run a backward pass and an Adam step.**

### Cited Findings
- **PyTorch reproducibility note (2.15 docs, last updated 14 May 2026):** "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds." It also says SDPA backends are not guaranteed to match bitwise because each one accumulates in a different order. — [PyTorch Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
- **Intel XPU (PyTorch 2.15 "Getting Started on Intel GPU", last updated 29 Sep 2026):** "Both training and inference workflows are supported. Both eager mode and torch.compile are supported." "FP32, BF16, FP16, and Automatic Mixed Precision (AMP) are all supported." Validated client hardware includes "Intel Core Ultra Processors with Intel Arc Graphics (CodeName: Meteor Lake-H)" on Windows 11, Ubuntu 24.04/26.04 and WSL2. Meteor Lake-H is this laptop's iGPU (Core Ultra 9 185H). — [PyTorch XPU](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
- **PyTorch AMP (2.15):** lists "XPU Op-Specific Behavior (Experimental)" and CPU ops that autocast to bfloat16 (conv, mm, bmm, linear, matmul, and others). — [torch.amp](https://docs.pytorch.org/docs/2.15/amp.html)
- **Apple MPS (PyTorch 2.15, last updated 12 Oct 2026):** "mps device enables high-performance training on GPU for macOS devices". "The MPS backend does not support torch.float64 … because Metal Shading Language has no double type." — [PyTorch MPS](https://docs.pytorch.org/docs/2.15/notes/mps.html)
- **DirectML (Microsoft Learn, last updated 8 Sep 2026):** "PyTorch with DirectML enables training and inference on DirectX 12-capable GPUs. PyTorch with DirectML is in public preview." Installed with `pip install torch-directml`. The page's example environment uses Python 3.10. — [PyTorch with DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)
- **AMD ROCm:** official PyTorch-on-ROCm install guide (wheels, Docker, build from source). It is Linux-first and points to separate Windows system requirements. — [ROCm PyTorch install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
- **JAX installation table (read 2026-11-11):**
  - Windows x86_64: CPU "yes", NVIDIA GPU "no", NVIDIA under WSL2 "experimental".
  - Google Cloud TPU: "yes" on Linux x86_64 only.
  - AMD GPU: "yes" on Linux x86_64.
  - Apple GPU: "experimental" on Mac aarch64.
  - Intel GPU: "experimental" on Linux x86_64, "no" on Windows and WSL2.
  - Install strings: `jax[tpu]`, `jax[cuda13]`, `jax[rocm7-local]`, `jax[oneapi]`.
  - [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- **Google TPU via PyTorch:**
  - The PyTorch/XLA README says: "Deprecated: This project is deprecated and replaced by TorchTPU. The PyTorch/XLA repo will be archived on November 15, 2026." — [pytorch/xla](https://github.com/pytorch/xla)
  - Google Cloud describes TorchTPU as "an open-source, PyTorch-native backend co-developed with the PyTorch team at Meta". It installs with `pip install torch_tpu` and uses `torch.device("tpu")`, eager by default, with torch.compile through OpenXLA/StableHLO and collectives via `tpu_dist`. It "supports standard PyTorch (ATen) operators in forward and backward passes." — [PyTorch on Cloud TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)
- **Intel NPU (OpenVINO 2025 "NPU Device" page):**
  - Describes "a low-power hardware solution, introduced with the Intel Core Ultra generation … to offload certain neural network computation tasks".
  - The page sits under "Running Inference → Inference Devices and Modes".
  - "Supported Inference Data Types … F32, F16 … U8. Computation precision for the HW is FP16." "The NPU device is currently supported by AUTO inference modes."
  - OpenVINO's only training-time feature is NNCF quantization-aware training, which is "applied to a PyTorch or TensorFlow training script" (training runs in that framework, not on the NPU).
  - [OpenVINO NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html); [OpenVINO training-time compression](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/compressing-models-during-training.html). Both were fetched with TLS verification disabled because of a local certificate-chain error.
- **Intel NPU Acceleration Library:** "This repository was archived by the owner on Apr 24, 2025", with an "End of Life Announcement" that redirects users to OpenVINO / OpenVINO GenAI. — [intel-npu-acceleration-library](https://github.com/intel/intel-npu-acceleration-library)
- **Qualcomm Hexagon:** the ONNX Runtime QNN EP offers backends `cpu`, `gpu` and `htp`, where `htp` "Offloads compute to NPU". The documentation covers quantized-model inference only. — [ORT QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- **AMD XDNA (Ryzen AI Software 1.9.0 docs):** "tools and runtime libraries for optimizing and deploying AI inference on AMD Ryzen AI powered PCs … on the NPU built in the AMD XDNA architecture". "The Ryzen AI development flow does not require any modifications to the existing model training processes." Training happens elsewhere. — [Ryzen AI docs](https://ryzenai.docs.amd.com/en/latest/)
- **Apple Core ML:** `MLUpdateTask` is "A task that updates a model with additional training data", and it applies to "updatable" Core ML models. The documentation does not say which compute unit (CPU, GPU or ANE) runs the update. — [MLUpdateTask](https://developer.apple.com/documentation/coreml/mlupdatetask)
- **Orion (arXiv 2603.06728, 2026):**
  - Claims "CoreML … do[es] not support on-device training" on the ANE.
  - Trains on the ANE only by "bypassing CoreML entirely via Apple's private _ANEClient and _ANECompiler APIs".
  - "Because the ANE bakes weights at compile time, naive training normally requires full recompilation per step (~4.2 s)", which they cut to 494 ms.
  - Trained "a 110M-parameter transformer on TinyStories for 1,000 steps in 22 minutes".
  - Conflict: Apple documents `MLUpdateTask` as on-device updating, so Orion's claim should be read as "no public ANE training path".
  - [arXiv 2603.06728](https://arxiv.org/abs/2603.06728)
- **Stepping Forward on the Last Mile (arXiv 2411.04036, 2024, Qualcomm authors):** "most existing low power neural processing engines (e.g., NPUs, DSPs, MCUs, etc.) are designed as fixed-point inference accelerators, without training capabilities". They propose fixed-point forward gradients instead of backprop. — [arXiv 2411.04036](https://arxiv.org/abs/2411.04036)
- **QScheduler (arXiv 2607.18802, IJCNN 2026):** "Zeroth-Order (ZO) optimization enables On-Device Learning (ODL) on NPU-equipped microcontrollers by estimating gradients through forward passes alone, bypassing the need for backpropagation primitives". — [arXiv 2607.18802](https://arxiv.org/abs/2607.18802)
- **ONNX Runtime on-device training:** an offline phase generates "training artifacts" on a server or desktop, and a training phase runs them on edge devices. The page does not list NPU execution providers for training. — [ORT On-Device Training](https://onnxruntime.ai/docs/get-started/training-on-device.html)
- **Huawei Ascend:** Ascend "NPUs" are datacentre training accelerators, a different class from laptop NPUs (e.g. TrainBridge arXiv 2610.29171; HyperParallel-MoE arXiv 2605.23764, abstracts read). They are not available to this project.

### Inferences
- "GPU → NPU migration of a training job" is not supportable on this laptop. There is no backward pass or Adam step on Intel AI Boost in any supported stack. Its documented compute precision is FP16 only, so fp32 m and v could not even be used natively. At most the NPU could serve as an **inference-only evaluator** of migrated weights, for example computing held-out loss through OpenVINO. That is a much weaker claim.
- "Laptop NPU" and "Ascend NPU" should never share a sentence in the paper without saying they are different device classes.

### Gaps
- No primary Apple page found that says whether `MLUpdateTask` ever runs on the ANE. The `model_personalization` doc JSON returned 404.
- The ORT on-device training EP support matrix was not found. The page that would have it returned 404.
- Qualcomm's own AI Engine Direct SDK docs were not fetched. The inference-only reading for Hexagon rests on the ORT QNN EP page and the Qualcomm-authored arXiv paper.

## 2. Free ways to run a few hundred training steps on a TPU, and loading PyTorch optimizer state there

### Takeaway
Kaggle offers free TPU time, about 20 hours a week and up to 9 hours per session according to the search-index text of Kaggle's own TPU doc. Third-party sources describe it as TPU v5e-8. Colab's FAQ confirms free TPUs but not the generation. For PyTorch on TPU the official path is now TorchTPU (`torch_tpu`, `device("tpu")`); PyTorch/XLA is archived on 15 Nov 2026. JAX/Optax is the alternative. Optax's Adam state is `ScaleByAdamState(count, mu, nu)`, which maps one-to-one onto PyTorch Adam's `(step, exp_avg, exp_avg_sq)`. No official converter between the two was found.

### Cited Findings
- **Colab FAQ:** "provides free of charge access to computing resources, including GPUs and TPUs". You select a TPU via Runtime > Change Runtime Type. The FAQ does not name the TPU generation. — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Kaggle TPU doc:** "You can use up to 20 hours per week of TPUs and up to 9h at a time in a single session". This is the search-engine snippet of the primary page, which renders client-side and returned only its title when fetched directly. — [Kaggle TPU docs](https://www.kaggle.com/docs/tpu)
- **Kaggle TPU generation:** described as "TPU VM v5e-8" with 16 GB HBM per chip by a user thread and a third-party page. UNVERIFIED against a Kaggle primary page. — [Kaggle discussion](https://www.kaggle.com/discussions/general/745817)
- **PyTorch/XLA README:** points to Kaggle notebooks (contrib/kaggle) for running PyTorch/XLA free on Kaggle. It also says the project is deprecated in favour of TorchTPU, with the archive date 15 Nov 2026. — [pytorch/xla](https://github.com/pytorch/xla)
- **TorchTPU:** `pip install torch_tpu`, eager by default, backward-pass ATen support, "preserves PyTorch tensor views and in-place mutations without a JAX runtime dependency". — [PyTorch on Cloud TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)
- **JAX on TPU:** `pip install -U "jax[tpu]"` on Linux TPU VMs. — [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- **Optax:** `scale_by_adam([b1, b2, eps, eps_root, ...])`; "ScaleByAdamState (count, mu, nu) State for the Adam algorithm." — [Optax transformations](https://optax.readthedocs.io/en/latest/api/transformations.html)
- **Google Cloud TPU Spot VMs:** "You also can't use 120-second preemption notice duration with TPU Spot VMs." — [TPU Spot VMs](https://cloud.google.com/tpu/docs/spot)
- **PyTorch DCP:** "handles load-time resharding which enables saving in one cluster topology and loading into another". It also has `torch_save_to_dcp` conversion. — [torch.distributed.checkpoint](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)

### Inferences
- minstate's flat, dtype-tagged wire format is framework-neutral bytes, so loading it on a TPU host is a deserialisation problem rather than a format problem. Under TorchTPU this should be decode-to-CPU-tensor then `.to("tpu")`. That is untested.
- Under JAX/Optax, three things need explicit handling and a unit test:
  - **Layout:** PyTorch `nn.Linear.weight` is (out, in); a Flax Dense kernel is (in, out). The moments must be transposed together with the weights.
  - **Step counter:** `step` maps to `count`.
  - **Epsilon placement:** Optax adds `eps` outside the square root and has an extra `eps_root` inside. Set `eps_root=0` to match PyTorch's default formula.
  - Without this, a "TPU migration" changes the optimizer, which breaks invariant 6.3. This is an inference from the documented signatures; verify it by comparing one update numerically on CPU before any TPU run.
- A JAX-side resume also changes the model code and kernels. It is a cross-framework migration and has to be labelled that way, not as "the same job on a TPU".

### Gaps
- Colab's free TPU generation and whether TorchTPU (`torch_tpu`) is preinstalled or installable on Kaggle or Colab TPU VMs: not confirmed.
- Whether Kaggle TPU access still requires phone verification: secondary sources only.

## 3. Numerical reproducibility across hardware: published measurements

### Takeaway
It is well established that bitwise equality across devices is not guaranteed and that tiny perturbations grow. Summers & Dinneen show that a one-bit change in initial parameters leads to "vastly different" final models. Implementation-level variance alone is measurable: up to 2.9% accuracy and 52.4% per-class accuracy differences (Pham et al.). Quantified cross-device **training-trajectory** gaps, meaning CPU vs GPU from identical state measured in loss over steps, are sparse in the literature. Most 2025–2026 cross-GPU numbers are for LLM inference. Some 2026 work shows bitwise cross-architecture reproducibility is achievable when the reduction order is fixed (Cooper et al., OPEN-1B), but only with purpose-built kernels.

### Cited Findings
- **Zhuang, Zhang, Song, Hooker (arXiv 2106.11872, 2021):** "top-line metrics such as top-1 accuracy are not noticeably impacted, [but] model performance on certain parts of the data distribution is far more sensitive". The cost of determinism "varies dramatically between … hardware types, e.g., with overhead up to 746%, 241%, and 196% on a spectrum of widely used GPU accelerator architectures". — [arXiv 2106.11872](https://arxiv.org/abs/2106.11872)
- **Pham et al. (ASE 2020, DOI 10.1145/3324884.3416545):**
  - "Even after excluding weak models, the accuracy difference is 10.8%."
  - "Implementation-level factors alone cause the accuracy difference across identical training runs to be up to 2.9%, the per-class accuracy difference to be up to 52.4%, and the training time difference to be up to 145.3%."
  - "83.8% of the 901 participants are unaware of or unsure about any implementation-level variance."
  - "only 19.5±3% of papers … use multiple identical training runs".
  - [Semantic Scholar record](https://www.semanticscholar.org/paper/1e5fb8003f866f9beeb1003f0be9a129d480ae75); [DOI](https://doi.org/10.1145/3324884.3416545)
- **Summers & Dinneen (ICML 2021, arXiv 2103.04514):** "all sources of nondeterminism have similar effects on measures of model diversity". "even one-bit changes in initial parameters result in models converging to vastly different values." — [arXiv 2103.04514](https://arxiv.org/abs/2103.04514)
- **Shanmugavelu et al. (arXiv 2408.05148, 2024):** studies the statistics of floating-point non-associativity in parallel programming models. It quantifies the performance cost of deterministic replacements for GPU atomics, examines PyTorch's deterministic options and the "reliability and completeness of the documentation", and evaluates Groq hardware determinism for inference. The abstract gives no headline number. — [arXiv 2408.05148](https://arxiv.org/abs/2408.05148)
- **Yuan et al. (arXiv 2506.09501, 2025), inference only:** "under bfloat16 precision with greedy decoding … DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size." The proposed fix, LayerCast, computes in FP32. — [arXiv 2506.09501](https://arxiv.org/abs/2506.09501)
- **Cooper, Jeong, Jeon, Young, Kim (arXiv 2609.25624, 2026), inference linear layers:** with a fixed, shape-determined reduction order and FP32 accumulation, "linear-layer outputs are bitwise identical across NVIDIA Ampere, Ada, and Hopper GPUs", running "1.17 to 3.1× faster end-to-end than the state-of-the-art solution". — [arXiv 2609.25624](https://arxiv.org/abs/2609.25624)
- **Donaghy et al., OPEN-1B (arXiv 2609.17380, 2026):** "this determinism does not carry across hardware". By imposing a definite order on GPU kernel reductions, batch ordering and collectives, they claim each training step can be replayed "on heterogeneous commodity hardware with bitwise certainty". The abstract gives no overhead figures. — [arXiv 2609.17380](https://arxiv.org/abs/2609.17380)

### Inferences
- For minstate, the honest null hypothesis is that **a device switch is itself a perturbation**. By Summers & Dinneen, that perturbation will separate the trajectories in parameter space. What matters is whether the held-out loss gap stays inside the seed band. This is exactly why a measured "device band" (Q6) is needed. Without it, any loss gap after a CUDA→CPU migration is uninterpretable.
- bf16 autocast behaves differently per device (separate CPU and XPU autocast op lists; XPU's is marked experimental). Mixed-precision settings will therefore not be identical across devices. Either run all cross-device cells in fp32 or report precision as a confound.

### Gaps
- No verified paper was found that reports **loss-trajectory divergence of the same training state continued on CPU vs GPU** with step-level numbers. Searches via arXiv returned none, and Semantic Scholar searches were rate-limited (HTTP 429). This is the gap minstate can fill at small scale.
- Schlögl et al., "Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks" (believed NeurIPS 2023): **UNVERIFIED**. arXiv title search returned nothing, DBLP was blocked by an anti-bot wall, and Semantic Scholar returned 429.

## 4. Prior systems that migrate or resume training across heterogeneous devices

### Takeaway
Prior "heterogeneous" training systems are heterogeneous across **GPU models**, not across device classes. They schedule (Gavel), partition (HetPipe, AutoHet) or preserve accuracy under elasticity (EasyScale), and spot-resilient systems (Bamboo, Varuna, Oobleck) assume homogeneous replicas. Checkpoint-portability work (UCP, ByteCheckpoint, PyTorch DCP) makes state portable across **parallelism layouts and cluster topologies**. None of the verified works migrates live Adam state between CPU, GPU, NPU and TPU and validates loss-trajectory equivalence against a seed band.

### Cited Findings
- **Gavel (Narayanan, Santhanam, Kazhamiaka, Phanishayee, Zaharia; arXiv 2008.09213, 2020):** heterogeneity-aware scheduling across "GPUs, TPUs, FPGAs, and custom ASICs". It improves "average job completion time and makespan by up to 3.5x". Gavel moves jobs between rounds by scheduling, not by state ablation. Venue (OSDI 2020) is not in the arXiv record: UNVERIFIED here. — [arXiv 2008.09213](https://arxiv.org/abs/2008.09213)
- **HetPipe (Park et al.; arXiv 2005.14038, 2020):** pipeline plus data parallelism on heterogeneous GPU clusters with "Wave Synchronous Parallel", with a "convergence proof of WSP". Models "converge up to 49% faster". Venue (USENIX ATC 2020): UNVERIFIED here. — [arXiv 2005.14038](https://arxiv.org/abs/2005.14038)
- **EasyScale (Li et al.; SC'23 per arXiv comment; arXiv 2208.14228):** "achieves consistent model accuracy under resource elasticity for both homogeneous and heterogeneous GPUs". It "traces the consistency-relevant factors carefully" and improves cluster utilisation "by 62.1%". This is the closest prior work on equivalence validation, but it covers GPU types only. — [arXiv 2208.14228](https://arxiv.org/abs/2208.14228)
- **AutoHet (Wang et al.; arXiv 2512.20953, 2025):** asymmetric 3D parallelism on heterogeneous spot GPUs. Its recovery "prioritizes to retrieve training states from local nodes, and only downloads the missing checkpoints from the cloud storage". Reported results: "1.79× speedup in training throughput" and "4.38× speedup of recovery speed". — [arXiv 2512.20953](https://arxiv.org/abs/2512.20953)
- **Bamboo (Thorpe et al.; arXiv 2204.12013, 2022):** redundant computation in pipeline bubbles to survive preemption. It "outperforms traditional checkpointing by 3.7x in training throughput, and reduces costs by 2.4x". — [arXiv 2204.12013](https://arxiv.org/abs/2204.12013)
- **Varuna (Athlur et al.; arXiv 2111.04007, 2021):** trains on "low-priority VMs that cost about 5x cheaper", including "a 200 billion parameter model". Venue (EuroSys 2022): UNVERIFIED here. — [arXiv 2111.04007](https://arxiv.org/abs/2111.04007)
- **Oobleck (Jang et al.; SOSP'23, DOI 10.1145/3600006.3613152; arXiv 2309.08125):** recovers from replicated model states across pipeline templates and "outperforms … Bamboo and Varuna by up to 29.6x". — [arXiv 2309.08125](https://arxiv.org/abs/2309.08125)
- **Universal Checkpointing (Lian et al.; arXiv 2406.18820, 2024):** "decoupling checkpoint structure from parallel training strategies and hardware configurations", with "negligible reconfiguration cost". — [arXiv 2406.18820](https://arxiv.org/abs/2406.18820)
- **ByteCheckpoint (Wan et al.; arXiv 2407.20143, 2024):** "parallelism-agnostic checkpoint representation that enables efficient load-time checkpoint resharding". It reduces runtime checkpoint stalls by "an average reduction of 54.20x". — [arXiv 2407.20143](https://arxiv.org/abs/2407.20143)
- **safetensors:** "a new simple format for storing tensors safely (as opposed to pickle) and that is still fast (zero-copy)". It stores tensors, not optimizer semantics. — [safetensors docs](https://huggingface.co/docs/safetensors/index)
- **ONNX Runtime on-device training:** exports training artifacts offline and trains on edge devices. It is a cross-runtime path, not a migration mechanism. — [ORT On-Device Training](https://onnxruntime.ai/docs/get-started/training-on-device.html)

### Inferences
- minstate's contribution can honestly be positioned this way: prior systems make state *portable* (layout and topology) or *schedulable* (heterogeneous GPUs). minstate measures *how much* of the state must cross and *what the destination device changes*, validated against a seed band. That is a measurement gap, not a systems gap.

### Gaps
- Sia (heterogeneity-aware scheduling, SOSP 2023), Metis (ATC 2024) and other heterogeneous-GPU training systems: not verified (arXiv title search found nothing).
- No cross-framework optimizer-state converter (PyTorch ↔ Optax) found in primary docs.

## 5. Bandwidth, notice windows, egress and interruption rates for the cost model

### Takeaway
The usable notice window varies by provider:

| Provider | Notice | Source wording |
|---|---|---|
| AWS | 2 minutes | best effort |
| GCP Spot VMs | 0 s by default plus a best-effort shutdown of up to 30 s, or 120 s if configured | 120 s is not available on TPU Spot |
| Azure | up to 30 s | best effort, via Scheduled Events |
| OCI | none stated | instances are "terminated" |

GPU instance networking ranges from 10–100 Gbps on single-GPU and L4-class instances to 3,200–3,600 Gbps on H100-class nodes. Internet egress is about $0.08–0.12 per GB after a free tier of 100 GB (AWS, Azure) or 1 GiB (GCP).

### Cited Findings
- **AWS:** "A Spot Instance interruption notice is a warning that is issued two minutes before Amazon EC2 stops or terminates your Spot Instance." "Interruption notices are emitted on a best effort basis." Hibernation gets no two-minute warning. — [AWS Spot interruption notices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html)
- **GCP Spot VMs:**
  - The preemption notice duration can be "120 seconds" or "0 seconds (default)".
  - "The shutdown period for Spot VMs is best effort and up to 30 seconds".
  - Preemption is signalled by the metadata `preempted` value turning TRUE, then ACPI G2 Soft Off.
  - [GCP Spot VMs](https://cloud.google.com/compute/docs/instances/spot)
- **GCP TPU Spot:** cannot use the 120-second notice. — [TPU Spot VMs](https://cloud.google.com/tpu/docs/spot)
- **Azure Spot:** Scheduled Events "are delivered on a best effort basis up to 30 seconds prior to the eviction". The Scheduled Events table gives "Preempt 30 seconds" as minimum notice. — [Azure Spot VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms); [Azure Scheduled Events](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/scheduled-events)
- **OCI preemptible:** "When preemptible capacity is reclaimed, the instance is terminated. Use the Events service to receive notifications". "can be terminated (deleted) at any time." No notice duration is stated. — [OCI Preemptible Instances](https://docs.oracle.com/en-us/iaas/Content/Compute/Concepts/preemptible.htm)
- **Instance bandwidth:**

  | Instance | Network | Source |
  |---|---|---|
  | AWS P5 / P5e / P5en (H100/H200) | "up to 3,200 Gbps of EFA networking" | [AWS P5](https://aws.amazon.com/ec2/instance-types/p5/) |
  | AWS P4d (A100) | "400 Gbps instance networking" | [AWS P4](https://aws.amazon.com/ec2/instance-types/p4/) |
  | AWS G6 (L4) | "up to 100 Gbps of network bandwidth" | [AWS G6](https://aws.amazon.com/ec2/instance-types/g6/) |
  | Azure ND H100 v5 | "3.2 Tbps of interconnect bandwidth per VM", "400 Gbps NVIDIA Quantum-2 CX7 InfiniBand" per GPU; front-end NIC max 80,000 Mbps | [Azure ND H100 v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndh100v5-series) |
  | GCP a3-highgpu-1g / 2g / 4g / 8g (H100) | 25 / 50 / 100 / 1,000 Gbps | [GCP GPU machine types](https://cloud.google.com/compute/docs/gpus) |
  | GCP a3-megagpu-8g (H100) | 1,800 Gbps | same |
  | GCP g2-standard-4 (1× L4) | 10 Gbps | same |

- **Egress:**
  - **AWS:** "AWS customers receive 100GB of data transfer out to the internet free each month". The tiered per-GB prices are rendered client-side and were not captured. — [AWS EC2 on-demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
  - **GCP:** cross-zone, same region $0.01/GiB; inter-region within North America $0.02/GiB; internet egress to North America $0.12/GiB from 1 GiB to 1,024 GiB, then $0.11 (first GiB free). — [GCP network pricing](https://cloud.google.com/vpc/network-pricing)
  - **Azure:** first 100 GB/month free; then $0.087/GB (next 10 TB, North America/Europe, Microsoft premium network); inter-region within North America or Europe $0.02/GB. — [Azure bandwidth pricing](https://azure.microsoft.com/en-us/pricing/details/bandwidth/)
- **Interruption statistics:**
  - **AWS Spot Advisor page:** "The average interruption frequency across all Regions and Instance types has historically been <5%"; savings "up to 90%". — [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
  - **AWS advisor data feed, read 2026-11-11:**
    - Linux type-region pairs by bucket: <5%: 6,780; 5–10%: 3,836; 10–15%: 2,190; 15–20%: 1,704; >20%: 4,843 (19,353 pairs).
    - GPU examples, us-east-1: p5.48xlarge <5% (57% savings); p4d.24xlarge 15–20% (56%); p3.2xlarge >20% (86%); g5.xlarge >20% (56%); g6.xlarge >20% (49%); p6-b200.48xlarge 10–15% (79%).
    - Many single-GPU G-family instances sit in the >20% bucket.
    - [spot-advisor-data.json](https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json)
  - **SpotLake (Lee, Hwang, Lee; IISWC 2022; arXiv 2202.02973):** a spot-dataset archive service. — [arXiv 2202.02973](https://arxiv.org/abs/2202.02973)

### Inferences
- **Drain budget.** With a 30 s window (Azure, GCP default) at 10 Gbps (≈1.25 GB/s, before protocol overhead), at most about 37 GB can leave. At 12 B/param (`full`) that is about 3B parameters; at 4 B/param (`w_only`) about 9B. On a 100 Gbps front end the limits rise by 10×. This is arithmetic from the cited figures, not a measurement.
- RDMA fabrics (EFA, InfiniBand) are intra-cluster. Migrating to a different provider or region uses front-end NICs and internet or inter-region egress pricing, so cross-region drain should use the front-end figures.

### Gaps
- AWS egress per-GB tiers (rendered client-side) and OCI egress pricing (HTTP 403): not captured.
- The AWS advisor's exact definition of "frequency of interruption" (the time window) was not captured from a primary page.
- GCP and Azure publish no comparable interruption-rate statistics, as far as was found.

## 6. Experimental design for a defensible single-laptop cross-device claim

### Takeaway
On this laptop, migration can honestly be measured among **CPU, CUDA (RTX 4050) and XPU (Arc iGPU, PyTorch 2.15 XPU in a separate venv)**, plus **TPU on a free Kaggle session** if time allows. The claim should be stated as "held-out loss after a cross-device resume stays within the same-device seed band, and the device-induced spread is X". Never claim bitwise equivalence. NPU and multi-node "GPU cluster" claims are out of reach and must not appear as results.

### Cited Findings
- PyTorch does not guarantee reproducibility across platforms or between CPU and GPU. — [PyTorch Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
- One-bit perturbations lead to divergent models, so the device switch must be treated as a perturbation with its own measured spread. — [Summers & Dinneen](https://arxiv.org/abs/2103.04514)
- Implementation-level variance is real and per-class effects are larger than top-line ones. Report per-tensor-class and held-out results, not only aggregates. — [Pham et al.](https://doi.org/10.1145/3324884.3416545); [Zhuang et al.](https://arxiv.org/abs/2106.11872)
- XPU on Meteor Lake-H supports training on Windows 11. — [PyTorch XPU](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
- The NPU is inference-only (FP16 compute). — [OpenVINO NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

### Inferences (proposed protocol)
1. **Bands, measured per scale and per T:**
   - **Seed band `eps_seed`**: 5 or more same-device controls differing only in seed (already in CLAUDE.md §5).
   - **Device band `eps_dev`**: one checkpoint at step T with the `full` arm, continued with the same seed and the same batches on each destination device (cpu, cuda, xpu). Compute the pairwise held-out-loss spread across devices over the recovery window.
   - **Same-device replay band `eps_rep`**: identical state continued twice on the same device, to measure run-to-run nondeterminism. For CUDA, with `torch.use_deterministic_algorithms(True)` this should be about 0, and that is a useful check.
2. **Migration cells:** cuda→cpu, cpu→cuda, cuda→xpu, xpu→cuda, cpu→xpu, each × arms {`full`, `w_v`, `w_m`, `w_only`} × protocol {`cold`, `frozen_warmup`}. Report StR and regret against the **same-device** control and against `eps_seed`. Report `eps_dev` alongside.
3. **What counts as "cross-device migration works":** for the `full` arm, the cross-device held-out-loss gap stays ≤ `eps_seed` for all j in the window, and `eps_dev` ≤ `eps_seed`. For reduced arms, the device does not change StR beyond what `eps_dev` explains: StR(arm, cross-device) − StR(arm, same-device) must be compared against the StR spread obtained by repeating the arm on the destination device.
4. **Controls that remove confounds:**
   - fp32 everywhere for cross-device cells. Autocast op lists differ per device and XPU autocast is experimental.
   - Identical wire bytes, checked by CRC before and after decoding.
   - Batch indices from a CPU generator (already done).
   - Dropout masks generated on CPU then moved to the device, otherwise device RNG streams differ by construction.
   - Log the torch version per venv. The CUDA venv is 2.14 and the XPU venv is 2.15, so cuda↔xpu cells change **both device and PyTorch version**. Either install one version in both venvs or label it as a confound.
5. **TPU (optional):** one cell, cuda→tpu with `full`, on a Kaggle TPU via `torch_tpu` (if installable) or via JAX/Optax with a verified state mapping (Q2). Label the JAX route "cross-framework".
6. **Bitwise-check tier (cheap and convincing):** after loading on the destination, compute one Adam update from a fixed batch on both devices and report the max-abs and relative difference per tensor class. This shows the size of the per-step device perturbation, which is the input to Summers-style divergence.

**What would count as overclaiming:**
- "GPU → NPU" or "migrates to NPUs" in any form. There is no training on the Intel AI Boost NPU.
- "GPU cluster" or multi-node results from one GPU, or from gloo processes on one machine. Call these "multi-process, single-host".
- "TPU" without an executed run, or a JAX run described as the same job.
- "Bitwise-equivalent" or "lossless" migration across devices.
- Any cross-device result reported without `eps_dev` and `eps_seed` next to it.
- Results from a single T or a single seed generalised to "works".
- Generalising from 2M–91M parameters to 7B without labelling it as extrapolation.
- Treating DirectML as a measured device without running it. It is in public preview and uses a separate torch-directml package.

### Gaps
- Whether torch-directml supports this machine's PyTorch versions was not checked. It was not installed, per instructions.
- Whether `torch_tpu` installs on Kaggle TPU VMs: untested.

## 7. Bottom line: device table, papers to cite, access log

### Takeaway
Cross-device claims are defensible for CPU, CUDA and XPU on this laptop, and for TPU only via a free Kaggle session. NPU training is not available in mainstream stacks.

### Cited Findings

**(a) Device classes**

| Device class | Training supported? | Backend (2026) | Measurable on this laptop? | Measurable on a free tier? |
|---|---|---|---|---|
| x86 CPU | Yes | PyTorch CPU; JAX CPU (Windows "yes") | Yes (Core Ultra 9 185H) | Yes (Colab/Kaggle CPU) |
| ARM CPU | Yes | PyTorch CPU; JAX Linux/Mac aarch64 | No | UNVERIFIED |
| NVIDIA CUDA | Yes | PyTorch CUDA; JAX CUDA (Linux; Windows "no", WSL2 "experimental") | Yes (RTX 4050 6 GB; currently busy) | Yes (Colab/Kaggle GPU) |
| AMD ROCm | Yes | PyTorch ROCm; JAX ROCm (Linux) | No (no AMD GPU) | No free tier found |
| Intel XPU (Arc / Data Center Max) | Yes | PyTorch XPU (validated on Meteor Lake-H, Windows 11); JAX Intel GPU experimental on Linux only | Yes (Arc iGPU, 2.15 venv) | No free tier found |
| Apple MPS | Yes (no float64) | PyTorch MPS; JAX Apple GPU experimental | No | No |
| DirectML | Claimed, public preview | torch-directml | Possible in principle (Arc or 4050); not installed or tested | n/a |
| Google TPU | Yes | JAX `jax[tpu]`; PyTorch via TorchTPU (`torch_tpu`); PyTorch/XLA archived 15 Nov 2026 | No | Yes: Kaggle TPU, ~20 h/week, ≤9 h/session (version v5e-8 UNVERIFIED); Colab TPU (generation UNVERIFIED) |
| Intel AI Boost NPU | **No** (inference only; FP16 compute) | OpenVINO NPU plugin; NPU Acceleration Library EOL Apr 2025 | Inference only | No |
| Qualcomm Hexagon NPU | **No** in mainstream stacks | ORT QNN EP (HTP), inference | No | No |
| Apple Neural Engine | **No public path**; research via private APIs (Orion) | Core ML (MLUpdateTask for updatable models, compute unit unspecified) | No | No |
| AMD XDNA NPU | **No** (inference only) | Ryzen AI Software 1.9 / Vitis AI EP | No | No |

**(b) Papers most worth citing**
1. Summers & Dinneen, ICML 2021 (arXiv 2103.04514). Justifies the device band: one-bit perturbations lead to divergent models.
2. Pham et al., ASE 2020 (DOI 10.1145/3324884.3416545). Quantifies implementation-level variance: 2.9% accuracy, 52.4% per-class.
3. Zhuang, Zhang, Song, Hooker, 2021 (arXiv 2106.11872). Tooling and hardware randomness, and the cost of determinism (up to 746%).
4. EasyScale, SC'23 (arXiv 2208.14228). The closest prior system that validates accuracy consistency across heterogeneous GPUs under elasticity.
   - Honourable mention: OPEN-1B (arXiv 2609.17380, 2026), the most recent claim of bitwise cross-hardware step replay.

**(c) Web access**
Web access worked. Sources that failed or were degraded:
- Semantic Scholar search: HTTP 429 after the first request.
- DBLP: HTTP 429 and an Anubis anti-bot wall.
- DuckDuckGo via ddgs: 4 of 5 queries timed out.
- docs.openvino.ai: local TLS chain error; fetched with verification disabled.
- Apple model_personalization JSON: 404.
- Kaggle docs: client-rendered; only the search snippet was used.
- Oracle networking pricing: HTTP 403.
- AWS egress tiers: client-rendered.
- ORT training API page: 404.
- github.com/google-pytorch/torchtpu: 404. The official TorchTPU page on docs.cloud.google.com worked.

### Inferences
- Of the author's intended claims, only CPU↔CPU, CPU↔GPU and GPU↔XPU (all single host) are measurable now. TPU is measurable with effort. "GPU cluster" requires multi-node hardware this project does not have. GPU↔NPU is not possible.

### Gaps
- UNVERIFIED and not cited as fact: Schlögl et al. (NeurIPS 2023); Sia (SOSP 2023); Metis (ATC 2024); the venues of Gavel (OSDI 2020), HetPipe (ATC 2020), Varuna (EuroSys 2022) and Zhuang et al. (MLSys 2022).

### BibTeX (verified papers)

```bibtex
@article{zhuang2021randomness,
  title   = {Randomness In Neural Network Training: Characterizing The Impact of Tooling},
  author  = {Zhuang, Donglin and Zhang, Xingyao and Song, Shuaiwen Leon and Hooker, Sara},
  journal = {arXiv preprint arXiv:2106.11872},
  year    = {2021},
  url     = {https://arxiv.org/abs/2106.11872}
}

@inproceedings{pham2020variance,
  title     = {Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance},
  author    = {Pham, Hung Viet and Qian, Shangshu and Wang, Jiannan and Lutellier, Thibaud and Rosenthal, Jonathan and Tan, Lin and Yu, Yaoliang and Nagappan, Nachiappan},
  booktitle = {Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year      = {2020},
  doi       = {10.1145/3324884.3416545}
}

@inproceedings{summers2021nondeterminism,
  title     = {Nondeterminism and Instability in Neural Network Optimization},
  author    = {Summers, Cecilia and Dinneen, Michael J.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2021},
  note      = {arXiv:2103.04514},
  url       = {https://arxiv.org/abs/2103.04514}
}

@article{shanmugavelu2024fpna,
  title   = {Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications},
  author  = {Shanmugavelu, Sanjif and Taillefumier, Mathieu and Culver, Christopher and Hernandez, Oscar and Coletti, Mark and Sedova, Ada},
  journal = {arXiv preprint arXiv:2408.05148},
  year    = {2024},
  url     = {https://arxiv.org/abs/2408.05148}
}

@article{yuan2025llmnondeterminism,
  title   = {Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference},
  author  = {Yuan, Jiayi and Li, Hao and Ding, Xinheng and Xie, Wenya and Li, Yu-Jhe and Zhao, Wentian and Wan, Kun and Shi, Jing and Hu, Xia and Liu, Zirui},
  journal = {arXiv preprint arXiv:2506.09501},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.09501}
}

@article{cooper2026crossarch,
  title   = {Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures},
  author  = {Cooper, Liam and Jeong, Shinnung and Jeon, Hyeran and Young, Jeffrey and Kim, Hyesoon},
  journal = {arXiv preprint arXiv:2609.25624},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.25624}
}

@article{donaghy2026open1b,
  title   = {OPEN-1B: A Fully Auditable Training Run},
  author  = {Donaghy, John and Wilcox, Brian and Ersoy, O{\u{g}}uzhan and Rastogi, Shikhar and St Arnaud, Adam and Titov, Alexey and Greenberg, Jordan and Fielding, Ben and Grieve, Harry},
  journal = {arXiv preprint arXiv:2609.17380},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.17380}
}

@article{narayanan2020gavel,
  title   = {Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads},
  author  = {Narayanan, Deepak and Santhanam, Keshav and Kazhamiaka, Fiodar and Phanishayee, Amar and Zaharia, Matei},
  journal = {arXiv preprint arXiv:2008.09213},
  year    = {2020},
  url     = {https://arxiv.org/abs/2008.09213}
}

@article{park2020hetpipe,
  title   = {HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism},
  author  = {Park, Jay H. and Yun, Gyeongchan and Yi, Chang M. and Nguyen, Nguyen T. and Lee, Seungmin and Choi, Jaesik and Noh, Sam H. and Choi, Young-ri},
  journal = {arXiv preprint arXiv:2005.14038},
  year    = {2020},
  url     = {https://arxiv.org/abs/2005.14038}
}

@inproceedings{li2023easyscale,
  title     = {EasyScale: Accuracy-consistent Elastic Training for Deep Learning},
  author    = {Li, Mingzhen and Xiao, Wencong and Sun, Biao and Zhao, Hanyu and Yang, Hailong and Ren, Shiru and Luan, Zhongzhi and Jia, Xianyan and Liu, Yi and Li, Yong and Lin, Wei and Qian, Depei},
  booktitle = {International Conference for High Performance Computing, Networking, Storage and Analysis (SC'23)},
  year      = {2023},
  note      = {arXiv:2208.14228},
  url       = {https://arxiv.org/abs/2208.14228}
}

@article{wang2025autohet,
  title   = {Diving into 3D Parallelism with Heterogeneous Spot Instance GPUs: Design and Implications},
  author  = {Wang, Yuxiao and Xu, Yuedong and Duan, Qingyang and Liu, Yuxuan and Jiao, Lei and Yu, Yinghao and Wu, Jun},
  journal = {arXiv preprint arXiv:2512.20953},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.20953}
}

@article{thorpe2022bamboo,
  title   = {Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs},
  author  = {Thorpe, John and Zhao, Pengzhan and Eyolfson, Jonathan and Qiao, Yifan and Jia, Zhihao and Zhang, Minjia and Netravali, Ravi and Xu, Guoqing Harry},
  journal = {arXiv preprint arXiv:2204.12013},
  year    = {2022},
  url     = {https://arxiv.org/abs/2204.12013}
}

@article{athlur2021varuna,
  title   = {Varuna: Scalable, Low-cost Training of Massive Deep Learning Models},
  author  = {Athlur, Sanjith and Saran, Nitika and Sivathanu, Muthian and Ramjee, Ramachandran and Kwatra, Nipun},
  journal = {arXiv preprint arXiv:2111.04007},
  year    = {2021},
  url     = {https://arxiv.org/abs/2111.04007}
}

@inproceedings{jang2023oobleck,
  title     = {Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates},
  author    = {Jang, Insu and Yang, Zhenning and Zhang, Zhen and Jin, Xin and Chowdhury, Mosharaf},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)},
  year      = {2023},
  doi       = {10.1145/3600006.3613152}
}

@article{lian2024ucp,
  title   = {Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism},
  author  = {Lian, Xinyu and Jacobs, Sam Ade and Kurilenko, Lev and Tanaka, Masahiro and Bekman, Stas and Ruwase, Olatunji and Zhang, Minjia},
  journal = {arXiv preprint arXiv:2406.18820},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.18820}
}

@article{wan2024bytecheckpoint,
  title   = {ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development},
  author  = {Wan, Borui and Han, Mingji and Sheng, Yiyao and Peng, Yanghua and Lin, Haibin and Zhang, Mofan and Lai, Zhichao and Yu, Menghan and Zhang, Junda and Song, Zuquan and Liu, Xin and Wu, Chuan},
  journal = {arXiv preprint arXiv:2407.20143},
  year    = {2024},
  url     = {https://arxiv.org/abs/2407.20143}
}

@article{kumaresan2026orion,
  title   = {Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference},
  author  = {Kumaresan, Ramchand},
  journal = {arXiv preprint arXiv:2603.06728},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.06728}
}

@article{feng2024steppingforward,
  title   = {Stepping Forward on the Last Mile},
  author  = {Feng, Chen and Zhuo, Shaojie and Zhang, Xiaopeng and Ramakrishnan, Ramchalam Kinattinkara and Yuan, Zhaocong and Li, Andrew Zou},
  journal = {arXiv preprint arXiv:2411.04036},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.04036}
}

@inproceedings{doamaral2026qscheduler,
  title     = {QScheduler: Adaptive Gradient Sampling for Zeroth-Order On-Device Training on INT8 NPUs},
  author    = {Do Amaral, Victor Felipe Domingues and Demaj, Pierre and Libessart, Erwan and Folliot, Laurent and Kolar, Anthony and B{\'e}nab{\`e}s, Philippe},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {arXiv:2607.18802},
  url       = {https://arxiv.org/abs/2607.18802}
}

@inproceedings{lee2022spotlake,
  title     = {SpotLake: Diverse Spot Instance Dataset Archive Service},
  author    = {Lee, Sungjae and Hwang, Jaeil and Lee, Kyungyong},
  booktitle = {IEEE International Symposium on Workload Characterization (IISWC)},
  year      = {2022},
  note      = {arXiv:2202.02973},
  url       = {https://arxiv.org/abs/2202.02973}
}
```
