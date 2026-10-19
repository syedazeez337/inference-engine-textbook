---
title: "Inference Engines: From First Principles to Serving Systems"
author: "DeepSeek research agents · reviewed and edited by Codex"
date: "Research cutoff: 19 October 2026"
lang: en
---



# About This Book {.unnumbered}

This textbook arranges the completed, reviewed research in this folder into a continuous study path. The underlying research and the reader implementation were delegated through CommandCode using **deepseek/deepseek-v4.1-flash-fast**. The chapters preserve primary citations and the source agents' stated access levels. The editor has applied the recorded corrections and converted mathematical notation for typesetting.

Read it in order for a first pass, or use the contents to revisit a subsystem. The goal is to connect a model's mathematical contract to its computation, state, hardware costs, scheduling, and measurement. The scope is a selective literature review through **19 October 2026**, not an exhaustive history or an independently reproduced set of benchmarks. Recommendations and future hypotheses are distinguished from established source findings.

::: {.definition}
**Reading the mathematics.** A parameter count and a hardware compute rate are different quantities. In inherited formulas, plain $P$ denotes a parameter count while $P_{\mathrm{peak}}$ denotes a rate in FLOP/s. In the teaching appendices these are written $P_w$ and $R_{\mathrm{peak}}$ to emphasize the distinction. Check the definitions and units before applying any equation.
:::

::: {.caution}
**Evidence and versions.** First-person statements about reading sources in the inherited material refer to the original research agents. Published speedups remain author-reported results tied to their original configurations. Implementation support is a snapshot at the research cutoff. The source inventory identifies reviewed reports, superseded drafts, audits, prompts and machine traces separately.
:::



# How to Study This Book {.unnumbered}

This is a selective, research-based study book about the forward execution of trained neural models, especially decoder-only Transformers. The literature cutoff is 19 October 2026. It is not an exhaustive survey and not a benchmark reproduction: it teaches mechanisms and derivations, does not report newly measured speedup figures, and treats every published result as a hypothesis to test on your own hardware. Source quality varies; flag claims resting only on abstracts or metadata. Distinguish source reports, cost-model derivations, and design recommendations.

Prerequisites: linear algebra, probability, calculus, Python, and unit conversion. No prior compiler or GPU experience is assumed, but you should be able to read pseudocode and run a small numerical experiment.

A workable route:

1. Foundations: threshold units, probabilistic language models, recurrence, attention, and the decoder-only Transformer. Track what state persists between steps.
2. Mechanics: computation graphs, tensor layouts, GEMM versus GEMV, and the separation of prefill from decode.
3. Cost model: parameter and FLOP formulas, KV-cache size, operational intensity, and the roofline. Rederive each formula by hand.
4. State: MHA, GQA, MQA, MLA, and recurrent or hybrid state. Compare capacity, bandwidth, and correctness.
5. Kernels and numeric formats: fusion, tiling, IO-aware attention, and quantization. Ask what changes bytes, what changes arithmetic, and what changes accuracy.
6. Serving: batching, scheduling, latency metrics, goodput, and exact speculative sampling. Keep kernel results separate from end-to-end results.
7. Build: a reference decoder, cache-versus-recompute tests, then profiling. Correctness before performance.

After each chapter, rederive the cost model, compute one example, and state which resource binds. When you meet a speedup claim, ask for hardware, workload, comparator, metric, and batch size; if any is missing, do not generalize it.

# Notation and Units {.unnumbered}

Units are part of the argument. $P_w$ is a parameter count, a dimensionless number of weights; $R_{\text{peak}}$ is a compute rate in FLOP/s. They are not interchangeable, and neither alone predicts latency. A dense forward pass costs about $2P_w$ FLOPs per token for its linear weights, but whether that arithmetic is the bottleneck depends on bytes moved and on $R_{\text{peak}}$ relative to bandwidth. Report FLOPs, bytes, and seconds separately.

| Symbol | Meaning | Unit |
|---|---|---|
| $L$ | decoder layers | count |
| $d$ | model width | elements |
| $h$ | query heads | count |
| $d_h$ | head width, $d/h$ | elements |
| $g$ | KV heads | count |
| $n$ | cached tokens | tokens |
| $T$ | prompt tokens | tokens |
| $B$ | batch size | sequences |
| $b$ | bytes per element | bytes |
| $P_w$ | parameter count | weights |
| $R_{\text{peak}}$ | peak compute rate | FLOP/s |
| $BW$ | memory bandwidth | bytes/s |
| $F$ | floating-point work | FLOPs |
| $D$ | bytes moved | bytes |
| $I$ | operational intensity, $F/D$ | FLOP/byte |
| $t$ | time | s |

Operational intensity is $I=F/D$ in FLOP/byte, and the ridge point is $I_{\text{ridge}}=R_{\text{peak}}/BW$. Below the ridge, bandwidth binds; above it, compute binds. For a phase with $F=100$ GFLOP, $D=20$ GB, $R_{\text{peak}}=100$ TFLOP/s, and $BW=1$ TB/s:

$$
t_{\text{compute}} \ge \frac{100\ \text{GFLOP}}{100\ \text{TFLOP/s}} = 1\ \text{ms}
$$

$$
t_{\text{memory}} \ge \frac{20\ \text{GB}}{1\ \text{TB/s}} = 20\ \text{ms}
$$

$$
t_{\text{floor}} \ge \max(1,20)\ \text{ms} = 20\ \text{ms}
$$

The compute floor is 1 ms, the memory floor is 20 ms, and the optimistic floor is their maximum, 20 ms. Intensity is 5 FLOP/byte against a ridge of 100 FLOP/byte, so memory binds. This assumes perfect overlap and ignores launch, dependency, and queueing costs; it is a lower bound inside the model, not a runtime prediction.

# Glossary {.unnumbered}

**Forward-only inference.** Evaluating a trained model without gradients or a backward pass; activations are freed after each layer.

**Autoregressive decoding.** Generating one token at a time, each conditioned on prior tokens.

**Prefill.** Parallel processing of prompt tokens; produces initial KV state and first logits.

**Decode.** Sequential token generation that reads and extends cached state.

**KV cache.** Stored keys and values per layer and token, avoiding recomputation during decode.

**MHA.** Multi-head attention: each query head has its own key and value head.

**GQA.** Grouped-query attention: query heads share a smaller number $g$ of KV heads.

**MQA.** Multi-query attention: all query heads share one KV head ($g=1$).

**MLA.** Multi-head latent attention: K and V are jointly compressed into a cached latent, with a decoupled rotary key.

**Operational intensity.** FLOPs per byte moved, $I=F/D$.

**Roofline.** A bound $\min(R_{\text{peak}}, I\cdot BW)$ relating attainable rate to intensity.

**Ridge point.** Intensity $R_{\text{peak}}/BW$ where compute and bandwidth bounds meet.

**GEMV.** Matrix-vector product; low reuse, usually memory-bound.

**GEMM.** Matrix-matrix product; tiling raises reuse and can become compute-bound.

**Batching.** Processing several sequences together to amortize weight traffic.

**Continuous batching.** Admitting and retiring requests at iteration granularity.

**Goodput.** Completed work meeting a latency target per unit time; SLO-goodput adds explicit TTFT and TPOT targets.

**TTFT.** Time from request arrival to first output token.

**TPOT / inter-token latency.** Time per output token after the first; report its distribution, not only its mean.

**Quantization.** Storing weights, activations, or cache in fewer bits; approximate and adds scale metadata.

**Weight-only quantization.** Shrinks weight bytes but not arithmetic; benefit falls as batching makes decode compute-bound.

**KV-cache quantization.** Compresses cached state; savings and error depend on head structure and positional encoding.

**Exact speculative sampling.** A draft proposes tokens; the target verifies them with a corrected acceptance rule that preserves the target distribution under stated assumptions, though not fixed-seed sequence identity.

**Verification.** Checking proposed tokens against target logits, then accepting a prefix and sampling a correction.

**MFU.** Model FLOPs utilization: achieved model FLOP/s divided by peak.



# From Neural Computation to Language Models

## Threshold logic and learning

McCulloch and Pitts (1943) modeled the neuron as an all-or-none threshold unit and showed that networks of such units can be described by propositional logic. The core calculus assumes a fixed net; learning appears only as an extension. In that framing, inference is evaluation of a logical circuit, not statistical prediction. ([paper](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf))

Rosenblatt (1958) introduced the perceptron as a probabilistic model of information storage and organization, with a learning procedure and predicted learning curves. This is the turn from hand-specified logic to parameters fitted from data. I did not verify the convergence theorem or full model. ([record](https://psycnet.apa.org/record/1959-09865-001))

Rumelhart, Hinton, and Williams (1986) is the widely cited modern statement of back-propagation: repeated weight adjustment to minimize output error, with hidden units learning task features. Caveat: the 1986 letter popularized the recipe; it was not the first derivation of reverse-mode differentiation. I did not verify the earlier lineage here. ([paper](https://www.nature.com/articles/323533a0))

## Probabilistic language models

Shannon (1951) framed next-symbol prediction as a way to estimate the entropy and redundancy of English, using n-gram approximations and human prediction experiments. This is a foundational milestone, not the sole origin of sequence modeling; Shannon's 1948 work already discusses language approximations. ([1951 paper](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), [1948 paper](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf))

Bengio et al. (2003) replaced sparse n-gram counts with a neural network over distributed word representations, explicitly to fight the curse of dimensionality. The model learns the joint probability of a sequence and generalizes to unseen sequences through similarity in representation space. Inference is a forward pass plus a normalization over the vocabulary. ([abstract](https://jmlr.org/papers/v3/bengio03a.html))

## Recurrence, encoder-decoder, attention

Hochreiter and Schmidhuber (1997) introduced LSTM to preserve error flow over long lags through a constant error carousel and multiplicative input/output gates. Update complexity is O(1) per time step and weight, and the recurrent state is fixed-size. That is a key inference property: memory does not grow with sequence length, but the state must compress the past. ([paper](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf))

Sutskever, Vinyals, and Le (2014) used a multilayered LSTM to map an input sequence to a fixed-dimensional vector and another LSTM to decode the target one token at a time. The fixed vector is an explicit bottleneck; decoding is inherently sequential. ([abstract](https://arxiv.org/abs/1409.3215))

Bahdanau, Cho, and Bengio (2015) removed that bottleneck by letting the decoder soft-search over encoder states. Attention scores are computed against all source positions, so inference cost grows with source length, but the decoder is no longer forced through one vector. ([abstract](https://arxiv.org/abs/1409.0473))

## Transformer and decoder-only inference

Vaswani et al. (2017) replaced recurrence with self-attention, enabling parallel training over sequence positions while keeping an encoder-decoder structure and a causal mask in the decoder. The architectural change is decisive for inference: training can process all tokens at once, but decoder inference still generates one token at a time. ([abstract](https://arxiv.org/abs/1706.03762))

GPT-1 (Radford et al., 2018) adapted the Transformer decoder into a 12-layer, 768-dimensional, 12-head decoder-only model trained with a next-token objective and then fine-tuned. GPT-3 (Brown et al., 2020) scaled the same autoregressive recipe to 175B parameters and demonstrated few-shot behavior without gradient updates. Once models are this large, inference cost and latency become first-order concerns. ([GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-3](https://arxiv.org/abs/2005.14165))

## Compressing the inference state

Shazeer (2019) observed that incremental decoding is limited by repeatedly loading key and value tensors, and proposed multi-query attention: all query heads share one key/value head. The KV cache shrinks by roughly the number of query heads, at some quality cost. ([paper](https://arxiv.org/abs/1911.02150))

Ainslie et al. (2023) generalized this to grouped-query attention, with an intermediate number of KV heads, and showed existing multi-head checkpoints can be uptrained into GQA. GQA is the common compromise: fewer KV heads than query heads, but more than one. ([paper](https://arxiv.org/abs/2305.13245))

DeepSeek-V2 (DeepSeek-AI, 2024) introduced multi-head latent attention: keys and values are jointly compressed into a latent vector, and a decoupled rotary key is cached separately. The paper gives the cache as (d_c + d_h^R) per token per layer, not 2× the latent. For DeepSeek-V2, d_c=512, d_h^R=64, d_h=128, n_h=128, L=60, giving 576 elements per token per layer and 34,560 elements per token — equivalent to GQA with 2.25 groups, as the paper states. ([paper](https://arxiv.org/abs/2405.04434))

Mamba (Gu and Dao, 2023) takes the other route: a selective state-space model with a fixed-size recurrent state, constant-time per-step inference, and no KV cache. The paper explicitly frames the tradeoff as state compression: attention stores the entire context; recurrent models compress it. ([paper](https://arxiv.org/abs/2312.00752))

Training and inference share the forward graph, but the surrounding computation differs.

- Training needs gradients. Backward roughly doubles matmul cost per token; a common estimate is 6PD FLOPs for P parameters and D tokens (2 forward, 4 backward). Inference is forward-only, about 2P FLOPs per token plus attention.
- Training parallelizes over tokens via teacher forcing; decoder inference cannot, because token t+1 depends on token t.
- Training stores activations for backward; inference frees them after each layer and keeps only the KV cache or recurrent state across steps.
- The bottleneck moves. Large-batch training is compute-bound; single-sequence decode is memory-bandwidth-bound because each token reads all weights and the whole KV cache.
- The objective changes: training minimizes loss; inference optimizes latency, throughput, memory capacity, and tail behavior.
- Precision changes: inference can quantize weights, activations, and KV cache; training is more sensitive to gradient precision and stability.

An inference engine is therefore not “training minus backward.” It is a stateful, latency-sensitive runtime whose cost is dominated by memory traffic and cache layout.

Classical symbolic inference operates on discrete symbols and rules: unification, resolution, forward/backward chaining, proof search. The computation is query-dependent; the graph of rule applications changes with the goal, and cost is governed by branching factor and search depth. Correctness is exact relative to the rule system.

Neural inference operates on dense tensors and learned weights. The graph is fixed, modulo dynamic shapes and control flow, and knowledge is distributed across parameters. Correctness is statistical and approximate. Cost is governed by FLOPs, bytes moved, and parallelism, not search depth. McCulloch and Pitts are the hinge: their threshold nets can realize propositional logic, so early neural computation looked like logic; learning and continuous activations turned the program from symbol manipulation into parameterized numeric evaluation. Modern LLM inference is still a computation graph, but a compiled and reused one — which is why compilers and kernel libraries matter.


# The Execution Graph and the Runtime

A trained model is a DAG of tensor operations, with weights as constants and activations as edges. Engines specialize it: fold constants, fuse elementwise ops, choose layouts, plan memory buffers, and dispatch kernels. TVM exposes graph-level and operator-level optimization, including fusion and auto-tuned schedules across backends. XLA compiles StableHLO/HLO with target-independent and target-specific fusion, shape specialization, buffer analysis, and partitioning. ONNX Runtime applies graph optimizations, partitions subgraphs across execution providers, and wraps each partition as a fused operator with a provider-specific allocator. ([TVM](https://arxiv.org/abs/1802.04799), [XLA](https://openxla.org/xla/architecture), [ONNX Runtime](https://onnxruntime.ai/docs/reference/high-level-design.html))

Two details matter for Transformers:

- **Tensor layout.** Activations are commonly [B, T, d]; weights [out, in]; KV cache [B, H, T, d_h] or [B, T, H, d_h]. The choice determines coalescing, transpose cost, and whether attention can read K/V contiguously. Layout assignment is a compiler decision, not cosmetic.
- **State.** Autoregressive decoding unrolls the same graph per token, with a growing KV cache as mutable state. The graph is static; the cache length is dynamic. Engines specialize on shape buckets or use dynamic-shape kernels, and must keep cache layout consistent across prefill and decode.

CPU and GPU execution differ mainly in the memory hierarchy. CPUs rely on caches, SIMD, and multithreading; GEMV is often bandwidth-bound, and weight quantization helps. GPUs have HBM plus on-chip SRAM and tensor cores; GEMM tiles well, but decode underutilizes compute and is sensitive to kernel-launch overhead and KV-cache traffic. The same roofline applies with different ridge points.


# Hardware, Arithmetic Intensity, and the Roofline

The roofline model bounds attainable performance by min(peak FLOP/s, bandwidth × operational intensity), where operational intensity is FLOPs per byte of DRAM traffic and the ridge point is peak FLOP/s divided by bandwidth. Williams, Waterman, and Patterson define intensity between caches and memory, not between processor and cache. ([paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf))

This gives the central split:

- **GEMV** (BLAS Level 2) computes y = Ax. For large A, each weight is used once per output element, so intensity is about 2/b FLOP/byte for b bytes per element. It is memory-bound.
- **GEMM** (BLAS Level 3) computes C = AB. Tiling reuses both operands, so intensity grows with tile size. Decode with batch B is a skinny GEMM; increasing B raises intensity until compute-bound.

Pope et al. (2023) name the phases: **prefill** processes all prompt tokens in parallel; **decode** generates output tokens autoregressively. They also state the baseline that an N-parameter decoder-only model needs 2N matmul FLOPs per token in the forward pass. Prefill is a large GEMM and can reach high model-FLOPs utilization; decode is a sequence of GEMV/skinny GEMMs dominated by loading weights and KV cache. ([paper](https://arxiv.org/abs/2211.05102))

FlashAttention is the clearest inference-relevant optimization that changes memory traffic rather than asymptotic FLOPs: it tiles attention to keep intermediates in on-chip SRAM and reduces HBM reads/writes, while remaining exact. It helps prefill (long attention matrices) and decode (KV-cache reads), but does not make attention free. ([paper](https://arxiv.org/abs/2205.14135))

Caveats: roofline is an upper bound, not a prediction; it ignores latency, occupancy, tail effects, and multi-chip communication. Attention FLOPs are often small relative to the MLP for large models, but the KV cache can dominate memory capacity and bandwidth.

Let \(L\) be layers, \(d\) model width, \(h\) query heads, \(d_h=d/h\), \(g\) KV heads, \(n\) cached tokens, \(T\) prompt tokens, \(B\) batch, and \(b\) bytes per element. A dense linear map of shape \(M\times K\) by \(K\times N\) costs approximately \(2MKN\) FLOPs, counting one multiply-add as two. For MHA with a non-gated two-matrix MLP of width \(4d\) and tied embeddings, parameters are approximately \(P\approx 12Ld^2+Vd\). This parameter formula does not apply unchanged to GQA, gated MLPs or MoE. Forward linear FLOPs are approximately \(2P\) per token when P denotes the active linear weights; embeddings and other excluded operations need separate accounting.

The roofline is:

\[
P_{\text{attain}}\leq\min\left(P_{\text{peak}},\; I\cdot BW\right),
\qquad
I=\frac{F}{D}\ \text{[FLOP/byte]},
\qquad
I_{\text{ridge}}=\frac{P_{\text{peak}}}{BW},
\]

with \(P_{\text{peak}}\) in FLOP/s, \(BW\) in bytes/s, and \(I\) in FLOP/byte. Equivalently, an idealized time floor is

\[
t \ge \max\left(\frac{F}{P_{\text{peak}}},\frac{D_{\text{HBM}}}{BW_{\text{HBM}}},\frac{D_{\text{net}}}{BW_{\text{net}}}\right)\ \text{[s]}.
\]

That is an **optimistic lower bound under assumed overlap**, not a promise. Critically, do **not** turn it into an unconditional real-runtime upper bound by adding compute time and transfer time. Dependencies, kernel launches, synchronization, occupancy, cache conflicts, queueing, retries, allocator fragmentation, and control-plane work can make real time exceed any idealized two-resource sum. The audit’s suggested \(T\le T_{\text{math}}+T_{\text{comms}}\) is only a bound inside the idealized roofline model; it is not a real-system upper bound ([roofline reference](https://jax-ml.github.io/scaling-book/roofline/)).

For a dense decoder:

\[
F_{\text{decode}}\approx 2P+4Lnd \quad \text{FLOP/token per sequence},
\]

\[
F_{\text{prefill}}\approx 2PT+4LT^2d \quad \text{FLOP for }T\text{ prompt tokens}.
\]

With batch \(B\) and equal context lengths, the linear-operation term becomes \(2PB\) and the attention term \(4LBnd\). Weight traffic can be amortized across the batch; the arithmetic still grows with B. Causal kernels may roughly halve the quadratic prefill attention term. Decode attention grows linearly with cached length; prefill attention grows quadratically with prompt length.

KV memory is:

\[
M_{\text{MHA}}=2LBndb,\qquad
M_{\text{GQA}}=2LBngd_hb,\qquad
M_{\text{MQA}}=2LBnd_hb,
\]

\[
M_{\text{MLA}}=LBn(d_c+d_R)b.
\]

Units are bytes; the factor 2 is keys plus values. MLA has no factor 2 because the latent \(c_{KV}\) jointly represents K and V and the decoupled RoPE key is shared. MQA is GQA with \(g=1\). Hybrid and recurrent models do not fit one formula: sum a per-layer state descriptor. For homogeneous Mamba-like SSM layers the core recurrent state is \(L B d_{\text{inner}}N b\) bytes independent of \(n\), plus convolution buffers and other implementation-specific state. Hybrid checkpoints also retain KV in their attention layers. Quantization changes \(b\) but adds scales, zero-points, packing, and metadata.

**Illustrative arithmetic, not a measurement:** 32 layers, 8 KV heads, \(d_h=128\), BF16 (\(b=2\)), 8,192 tokens gives \(2\cdot32\cdot8\cdot128\cdot2\cdot8192=1{,}073{,}741{,}824\) bytes, or 1 GiB per request. Sixteen such requests need 16 GiB for KV alone. An 8B BF16 checkpoint is about 16 GB raw; 4-bit weights are about 4 GB raw before scales and workspaces.


# Deriving Parameters, FLOPs, and State Memory

## Assumptions

L layers; model width d; h query heads; d_h = d/h; g KV heads; vocabulary V; d_ff = 4d; b bytes per element (2 for FP16/BF16); batch B; context n; prompt length T. Ignore biases, layer norms, RoPE, softmax, sampling, padding, and allocator overhead. Count one multiply-add as 2 FLOPs.

## Parameters and FLOPs

For the MHA special case with a non-gated 4d-width MLP, per-layer attention projections Q, K, V, O = 4d²; MLP = 2·d·4d = 8d²; total = 12d². So:

$P \approx 12Ld^2+Vd$ (tied embeddings; untied adds another Vd).

More generally for ordinary GQA/MQA projections and an m-matrix MLP, $P \approx L(2d^2+2gd^2/h+m d d_{\mathrm{ff}})+Vd$, where m=2 for a non-gated MLP and m=3 for a gated MLP. MLA and MoE need their own parameter accounting. ([GLU variants](https://arxiv.org/abs/2002.05202))

For GPT-3, L=96, d=12288: 12·96·12288² ≈ 173.9B, plus embeddings ≈ 174.6B, consistent with the reported 175B. ([paper](https://arxiv.org/abs/2001.08361))

Forward FLOPs per token: $2P$ for matmuls, plus attention:
- Decode at context n: $4Lnd$ (QKᵀ and AV, each 2nd per layer).
- Prefill T tokens: $4LT^2d$ dense; causal kernels can roughly halve the score/AV term.

Training, for D tokens: $C_{\mathrm{train}}\approx6PD$ (forward 2P, backward 4P per token). This is a derivation consistent with Kaplan et al.’s compute-scaling framing, not a measured constant.

## KV cache

For MHA, each token per layer stores K and V, each of width h·d_h = d:

$$M_{\mathrm{MHA}}=2Lndb\quad\text{bytes per sequence}.$$ 

For GQA with g KV heads:

$$M_{\mathrm{GQA}}=2Lngd_hb\quad\text{bytes per sequence}.$$  Ratio to MHA = g/h.

For MQA, g=1: $M_{\mathrm{MQA}}=2Lnd_h\quad\text{elements per sequence}$; ratio = 1/h.

For MLA, cache the compressed latent and the decoupled rotary key:

$$M_{\mathrm{MLA}}=(d_c+d_R)Ln\quad\text{elements per sequence}.$$ 

No factor of 2: the latent jointly represents K and V; the rotary key is shared across heads. DeepSeek-V2: d_c=512, d_R=64, L=60, so 576·60 = **34,560 elements per token**; at 2 bytes, **69,120 bytes per token**. This is linear in n, not constant.

For a state-space model, cache the recurrent state, not K/V:

$M_{\mathrm{state}}=L d_{\mathrm{inner}}N\quad\text{elements}$ describes the core recurrent state in this simplified model, independent of n. Actual implementations also retain convolution buffers and possibly checkpoint/replay state; obtain their shapes and dtypes from the checkpoint and runtime rather than treating this expression as total runtime memory.

Batch multiplies all of these; precision scales bytes by b; padding and allocator fragmentation are extra.

## Hand-checkable example

Take L=32, d=4096, h=32, d_h=128, n=4096, b=2, d_ff=4d.

- P ≈ 12·32·4096² = **6.442B** parameters; weights = **12.885 GB**.
- MHA KV = 2·32·4096·4096·2 = **2.147 GB** (2 GiB).
- GQA with g=8: 2·32·4096·8·128·2 = **0.537 GB** (512 MiB), 4× smaller.
- MQA: **0.067 GB** (64 MiB), 32× smaller.
- MLA with d_c=512, d_R=64: 576·32·4096·2 = **0.151 GB** (144 MiB), 14.2× smaller than MHA; equivalent to GQA with 2.25 groups.
- Decode FLOPs = 2P + 4Lnd = 12.885 + 2.147 = **15.03 GFLOP per token**.
- Decode bytes = weights + MHA KV = **15.03 GB**; operational intensity ≈ **1.0 FLOP/byte**.
- Prefill at T=4096: 2PT + 4LT²d = 52.78 + 8.80 = **61.57 TFLOP**; intensity against weights+KV ≈ **4,096 FLOP/byte**.

On a hypothetical machine with 300 TFLOP/s FP16 peak and 2 TB/s HBM, the ridge point is 150 FLOP/byte. The decode estimate is far below it. The prefill estimate omits activation and intermediate traffic, so its ~4,096 FLOP/byte is optimistic and does not prove prefill is compute-bound. Fusion, shapes and measured traffic determine that conclusion. A bandwidth-only floor for the assumed batch-1 decode traffic is 15.03 GB / 2 TB/s ≈ 7.5 ms/token. These are illustrative calculations, not benchmarks; the example also excludes vocabulary embeddings/output projection from its P value.

Batching amortizes weight traffic, not FLOPs. For equal context lengths, decode-step FLOPs are approximately B(2P+4Lnd), while bytes in this idealized model are b(P+2BLngd_h), plus activations and overhead. Intensity increases while weight traffic dominates; growing KV traffic can keep decode bandwidth-bound even at large batch.


# Attention Kernels and Memory Traffic

**FlashAttention (2022)** reframed attention as an I/O problem. It tiles Q/K/V through SRAM, uses online softmax, and recomputes attention in the backward pass instead of materializing the N×N score matrix. The paper argues Θ(N²d²/M) HBM accesses versus Θ(Nd+N²) for standard attention, with linear extra memory and exact output [1](#ref-systems-1). This is a training-era kernel/algorithm contribution; it says nothing about paged KV, continuous batching, or decode.

**FlashAttention-2 (2023)** fixed work partitioning: fewer non-matmul FLOPs, parallelism over sequence length as well as batch/heads, and warp-level splitting that avoids split-K communication. On A100 it reports roughly 2× over FA1 and 50–73% of theoretical peak [2](#ref-systems-2). It remains synchronous and Ampere-oriented.

**FlashAttention-3 (2024)** is Hopper-specific: warp-specialized producer/consumer pipelines, asynchronous WGMMA and TMA, GEMM–softmax overlap, and FP8 with block quantization plus incoherent processing. On H100 SXM5, FP16, sequence lengths 512–16k at 16k total tokens and head dimensions 64/128/256, it reports 1.5–2.0× over FA2 and up to 740 TFLOP/s (75% utilization); FP8 approaches 1.2 PFLOP/s with lower error than per-tensor scaling [3](#ref-systems-3). Caveats: H100-only, attention-only, and FP8 parity depends on head dimension and causal masking.

**FlashAttention-4 (2026)** addresses Blackwell’s asymmetric scaling: tensor-core throughput doubled while shared-memory bandwidth and exponential-unit throughput did not, so softmax and shared-memory traffic dominate. It uses fully asynchronous MMA, larger tiles, TMEM, 2-CTA MMA, software-emulated exponentials, conditional softmax rescaling, and a CuTe-DSL implementation. On B200, BF16, 1k–32k sequences at 32k total tokens, head dimensions 64/128/(192,128), it reports up to 1.3× over cuDNN 9.13, 2.7× over Triton, and 1613 TFLOP/s (71%) [4](#ref-systems-4). Two caveats: the appendix names a B100 180GB benchmark platform while the abstract says B200, and newer cuDNN versions have absorbed some techniques. Code is released, but I found no verification that vLLM, SGLang, or TensorRT-LLM ship FA4 as a default backend.

**FlashInfer (MLSys 2025)** is the serving-oriented complement: a unified block-sparse-row layout covering paged, ragged, tree, and block-sparse KV; composable formats for shared prefixes; JIT-generated attention variants; and a load-balanced scheduler compatible with CUDA graphs. It is integrated into SGLang, vLLM, and MLC-Engine [5](#ref-systems-5). On SGLang v0.3.4, Llama 3.1 8B/70B, A100/H100, under a P99 TTFT target, it reports 29–69% lower inter-token latency than the Triton backend, plus 28–30% lower long-context latency; sparse gather costs about 10% on prefill. These are version- and integration-specific results.

**Derivation.** Kernel peak is a weak predictor of serving throughput. Prefill can be compute/softmax-bound and benefits from FA3/FA4-class pipelines; ordinary small-batch decode is often bandwidth-bound and has query length one, and depends more on paged/varlen addressing, split-K/FlashDecoding work partitioning, launch overhead, and CUDA-graph stability. PagedAttention’s indirection measured 20–26% higher attention-kernel latency than a contiguous kernel while still winning end-to-end by enabling larger batches [7](#ref-systems-7).


# Continuous Batching, Paging, and Prefix Reuse

**Orca (OSDI 2022)** introduced iteration-level scheduling and selective batching. The scheduler invokes the engine one iteration at a time, admits arrivals and retires finished requests between iterations, batches non-attention operations, splits attention per request, and reserves KV slots up to `max_tokens`. It uses FCFS with no preemption. On GPT-3 175B, 16×A100-80GB, with a synthetic Poisson trace, it reports 36.9× throughput over FasterTransformer at matched median normalized latency [6](#ref-systems-6). The number is tied to an obsolete baseline; the durable contribution is the scheduling abstraction.

**vLLM/PagedAttention (SOSP 2023)** replaced contiguous per-request KV allocation with fixed-size blocks and a logical-to-physical block table, enabling on-demand allocation, copy-on-write sharing for parallel sampling/beam search, and prefix sharing. The paper reports that prior systems used only 20.4–38.2% of KV memory for token states; block size 16 is the default compromise. It reports 2–4× throughput over Orca/FasterTransformer on OPT/LLaMA with ShareGPT/Alpaca traces at similar latency [7](#ref-systems-7). The kernel itself is 20–26% slower than a contiguous optimized kernel—the price of memory flexibility.

**vLLM V1 (2025–2026)** is a re-architecture. The scheduler treats prompt and output tokens uniformly as a dictionary `{request_id: num_tokens}`, making chunked prefill, prefix caching, and speculative decoding variants of one token-budget mechanism. Chunked prefill is default whenever possible; prefix caching is default with near-zero overhead at a 0% hit rate in the project’s benchmark; persistent batches, incremental TP diffs, `torch.compile`, and piecewise CUDA graphs cut CPU overhead. V0 is deprecated and default preemption is recompute-oriented; explicit KV offload connectors are a separate capability, and and memory pressure raises tail latency through recomputation rather than swap [16](#ref-systems-16)[17](#ref-systems-17).

**SGLang/RadixAttention (2023–2024)** treats KV as a radix-tree LRU cache across requests and calls, with longest-shared-prefix-first cache-aware scheduling and frontend fork/join hints. It reports up to 6.4× throughput on prefix-heavy LM-program workloads on A10G/A100, with radix-management overhead below 0.3%; the vLLM baseline was v0.2.5, so the comparison is dated [8](#ref-systems-8). Production Chatbot Arena hit rates were 52.4% (LLaVA-NeXT-34B) and 74.1% (Vicuna-33B). The paper itself notes that greedy cache-aware scheduling can starve requests and that its DFS optimality result is offline and assumes cache capacity at least the maximum request length.

**SGLang HiCache (2025–2026)** extends this to L1 GPU, L2 host DRAM, and L3 distributed storage (Mooncake, HF3FS/3FS, NIXL, AIBrix, LMCache, file). HiRadixTree records which tier holds each span; prefetch policies are `best_effort`, `wait_complete`, and `timeout`; write-back is write-through, selective, or on-eviction. L2 is node-local and instance-private, so cross-instance reuse requires L3. Under replicated MLA KV layouts, write-back must account for identical state on TP ranks; DP, DCP and other sharding layouts can change this. Cache size and hit rate are nonlinear, and page size trades metadata efficiency against partial-page hits [19](#ref-systems-19).

**Derivation.** These are complementary layers: Orca is the scheduling abstraction, vLLM the memory allocator, SGLang the reuse policy. Prefix caching pays only when routing preserves affinity or a shared L3 exists; otherwise the cache is cold and its metadata is pure overhead.


# Scheduling and Disaggregated Serving

**Sarathi-Serve (OSDI 2024)** names the central tradeoff: prefill-prioritizing schedulers cause “generation stalls” and TBT spikes; decode-prioritizing schedulers waste throughput. It chunks prefills and builds stall-free hybrid batches under a token budget derived from the TBT SLO, so new prefills piggyback on decode batches without pausing them. Versus vLLM under stated P99 TBT SLOs, it reports 2.6× capacity for Mistral-7B on one A100, 3.7× for Yi-34B on two A100s, and 5.6× for Falcon-180B with TP4-PP2 on eight A100s [9](#ref-systems-9). Caveats: chunking adds up to ~25% prefill overhead at chunk 512; tile quantization made chunk 257 ~32% slower than 256 in their example; and the artifact is a research fork, not feature-complete vLLM.

**DistServe (OSDI 2024)** makes the goodput argument: prefill is compute-bound and TTFT-sensitive, decode is memory-bound and TPOT-sensitive, and colocation couples their parallelism and creates interference. It assigns separate pools, searches per-phase TP/PP and replica counts, and uses pull-based KV transfer. On OPT-13B/66B/175B, A100-80GB, a 25 Gbps cross-node testbed, Poisson arrivals, and 90% attainment, it reports 7.4× higher request rate or 12.6× tighter SLO versus vLLM/DeepSpeed-MII [10](#ref-systems-10). Limits: FCFS, no preemption or fault tolerance, and KV transfer below 0.1% of latency only with node-affinity placement and NVLink. The working file cites both an OSDI PDF and arXiv:2401.09670; I treat them as the same paper.

**Splitwise (ISCA 2024)** broadens the argument to cluster design: prompt machines are compute-heavy, token machines memory/power-heavy, and a mixed pool absorbs fragmentation; layer-wise KV transfer overlaps with prompt computation. Using Azure production traces, an event-driven simulator, and two-machine vLLM experiments on A100/H100, it reports throughput/cost gains, but the cluster-level results are simulated and the headline cost comparison is not fully specified [11](#ref-systems-11). Heterogeneous pools add operational complexity.

**Mooncake (FAST 2025)** is the production counterpoint. Kimi separates prefill and decode clusters but also pools CPU DRAM, SSD, and RDMA NICs into a distributed KV cache; a KVCache-aware Conductor schedules by prefix locality and load; chunked pipeline parallelism scales long-context prefill. On 16×8 A800 nodes with real traces it reports 59–498% higher effective request capacity; in production A800/H800 clusters, 115%/107% more requests at over 100B tokens/day. It recommends ≥100 Gbps for the global cache and rejects requests with HTTP 429 when SLOs cannot be met [12](#ref-systems-12). Caveats: proprietary traces and model; long-context-heavy workload; a dummy LLaMA3-70B is used for reproducibility.

**AMPD (February 2026; v2 July 2026)** targets multi-round/agentic traffic where incremental prefills interleave with decode. It adaptively runs each prefill locally on the decode worker (no transfer, but pausing decode) or remotely on a prefill worker, reorders the prefill queue, and solves an ILP for deployment. Built on Dynamo/NIXL, on four 8×H20-96GB servers with 200 Gbps IB and Llama3.1-70B/Qwen3-32B/Mixtral-8x7B, it reports 67.29–339.74% average SLO-attainment improvement over colocated and disaggregated baselines [15](#ref-systems-15). This is a paper prototype, and the metric is trace-specific SLO attainment, not universal throughput.

**Derivation.** vLLM positions its disaggregated-prefill implementation for TTFT/ITL isolation rather than raw-throughput improvement [18](#ref-systems-18). This is not a universal verdict: DistServe and Mooncake study SLO-constrained goodput/effective capacity. Disaggregation can win or lose depending on interference, placement, transfer and load. Measure raw throughput and goodput separately at equal resource budgets.


# Parallelism, Collectives, and Expert Placement

**Tensor parallelism (TP)** shards each layer and needs an all-reduce per attention/FFN block; it is latency-efficient inside an NVLink domain and poor across commodity nodes. **Pipeline parallelism (PP)** shards layers with point-to-point activations, scales capacity across nodes, and creates bubbles; Sarathi’s uniform token budgets reduce them, and TensorRT-LLM’s in-flight batching requires context-phase sequences to precede generation-phase sequences in the packed input tensor [20](#ref-systems-20). **Data parallelism (DP)** replicates the model; the hard part is routing. KV-cache-aware routing (Dynamo Smart Router, llm-d, Mooncake Conductor) trades prefix affinity against load, and round-robin routing destroys hit rates while pure affinity overloads hot replicas.

**Expert parallelism (EP)** is now first-class for MoE. DeepSeek-V3 (671B total, 37B active, MLA, one shared plus 256 routed experts, 8 active per token, at most 4 nodes per token) uses TP4 + sequence parallelism, with DP8/EP32 for prefill (4 nodes, 32 GPUs) and DP80/EP320 for decode (40 nodes, 320 GPUs), plus redundant experts rebalanced roughly every 10 minutes [13](#ref-systems-13). Cross-node EP in training was near 1:1 compute-to-communication; inference all-to-all is the dominant constraint. This is a model-specific recipe with a 40-node minimum decode unit, not a default.

**Context parallelism (CP)** splits the sequence. Ring Attention distributes query blocks and rotates KV blocks around a ring, overlapping transfer with blockwise attention/FFN; context length scales linearly with devices, with block size constrained by FLOPS/bandwidth [14](#ref-systems-14). The paper reports 32M tokens on 32 A100s and 131M on TPUv4-512, but its experiments are training. CP is mainly a long-prefill TTFT tool; decode CP is harder because KV sharding adds per-token communication and interacts badly with attention DP.

**Collectives.** NCCL remains the default for TP/PP/EP; KV transfer uses NIXL/UCX/RDMA or vendor engines (Mooncake, Ascend). Orca’s separation of control-plane gRPC from NCCL data-plane remains a useful rule: control messages should not serialize on GPU collectives [6](#ref-systems-6).


# Quantization and Numerical Contracts

The useful framing is not a chronology of “weights, then everything else,” but three interacting budgets: **weight/activation bytes**, **KV/state residency**, and **step/scheduling cost**. A low-bit weight model can remain KV-bound at long context; a speculative decoder can hide draft latency but cannot create KV capacity; and an agent workload is decode-dominated only while its cached state survives.

The drafts’ claim that 2022–2024 was “mostly about weights” is too narrow. KV-cache quantization, latent attention, tree verification, and serving scheduling were already first-class concerns in the reviewed sources. Weight quantization was the most standardized and easiest to benchmark, not the only active problem. Attention layout, state growth, and request scheduling already constrained real deployments.

The second axis is correctness. Correct rejection-sampled speculation and compatible exact prefix reuse preserve target semantics subject to numerical limits. Weight/KV quantization and approximate acceptance change the deployed target or sampling rule. Native sparse, linear-attention and diffusion checkpoints are alternative architectures, not automatically approximate executions of a dense autoregressive checkpoint. Their runtime can faithfully execute their own semantics; comparisons require separate task-quality evaluation.

**GPTQ** treats layer-wise weight quantization as a second-order reconstruction problem. It quantizes weights in blocks, uses an approximate Hessian inverse, and updates remaining weights to compensate for rounding error. Its limitation is structural: it is weight-only, reduces memory traffic rather than arithmetic, does not quantize activations, and does not improve compute-bound batched execution. Its quality claims are calibration- and model-specific; lower bit-widths require smaller groups and still degrade.

**AWQ** identifies salient channels by activation magnitude rather than weight magnitude and protects them with per-channel scaling. It avoids backpropagation and reconstruction, which the authors argue reduces calibration overfitting. It remains W4A16 weight-only: speed depends on packing and fused dequantization kernels, and it does not make arithmetic cheaper. Its robustness advantage is relative, not universal.

**SmoothQuant** addresses the activation-outlier problem that blocks W8A8. Outliers persist in fixed channels, so per-tensor activation quantization wastes effective bits. SmoothQuant applies a mathematically equivalent per-channel scaling that migrates quantization difficulty from activations to weights. It preserves accuracy on several large models under INT8 GEMM, but it is an 8-bit solution tied to per-tensor static quantization; it is not a path to 4-bit activations.

**FP8** defines E4M3 and E5M2 with different range/precision tradeoffs and recommends E4M3 for weights and activations, E5M2 for gradients. The original work was largely simulated, not a hardware speed demonstration. DeepSeek-V3 is stronger production evidence: a large MoE trained in FP8 with fine-grained scaling and frequent FP32 promotion because H800 accumulation retained limited precision. That is a co-designed hardware/software result, not proof that naive FP8 works everywhere. FP8 inference still depends on tensor-core generation, scaling granularity, and accumulation behavior.

**FP4** is more aggressively hardware-specific. NVFP4 uses E2M1 elements, small micro-blocks, per-block E4M3 scales, and a per-tensor FP32 scale. NVIDIA’s pretraining result required random Hadamard transforms, two-dimensional weight scaling, stochastic rounding on gradients, and keeping a minority of linear layers in BF16; late BF16 switching closed most of the remaining loss gap. PTQ works better on very large models than on small or sensitive post-trained models, where quantization-aware distillation is often needed. QAD is a training procedure, not a free conversion. All of these are vendor technical reports, not independent reproductions.

**KV quantization** has a different failure surface. KIVI observes that key caches have persistent per-channel outliers while value caches do not, so it quantizes keys per-channel and values per-token and keeps a recent FP16 residual window. It degrades on multi-query-attention models, which may need higher precision. KVQuant adds pre-RoPE key quantization, non-uniform datatypes, sparse outlier isolation, and attention-sink handling; it requires calibration, custom kernels, and fused RoPE. MLA changes the outlier structure entirely, so per-channel assumptions do not automatically transfer. DeepSeek-V2 already deploys MLA with roughly 6-bit average KV quantization.

The central correction is that **quantization changes the target distribution**. Exact speculative decoding does not undo quantization error: it verifies the quantized model’s distribution. Quality deltas are small in many reported settings but nonzero, and they are largest on reasoning, long context, small models, and architectures whose outlier structure differs from the calibration assumptions.


# Speculative Decoding and Verification

Leviathan et al. and Chen et al. independently introduced draft-and-verify decoding. A draft proposes tokens; the target scores them in parallel; a modified rejection rule accepts a prefix and resamples from the residual distribution. The guarantee is **distribution preservation**, not sample-for-sample identity. Under stochastic sampling, exact speculative decoding produces different random samples that follow the same target distribution; “identical outputs” language applies only to greedy decoding or within hardware numerics. The drafts’ phrasing on this point was imprecise.

Exact speculation often targets latency, but can also improve throughput in suitable regimes. Its benefit depends on acceptance, target/draft cost, verification work, batch size and scheduling. At high load, additional verification can consume scarce compute and reduce goodput; this must be measured rather than assumed for every large batch.

**Medusa** replaces the separate draft model with extra decoding heads and tree attention. Its standard rejection mode can be exact, but its default **typical acceptance** selects “reasonable” candidates using an entropy threshold and is explicitly not distribution-preserving. That is an approximate quality/speed trade, not lossless acceleration. Tree size has diminishing returns, and at sufficiently large batch the linear layers become compute-bound.

**EAGLE-3** predicts tokens directly, fuses low-, mid-, and high-level target features, and simulates multi-step drafting during training. It uses strict speculative acceptance, so it is lossless with respect to its target, but it requires training a draft per target and its gains are task- and benchmark-specific.

The 2026 development is **Speculative Speculative Decoding (SSD/Saguaro)**. It places the draft on a separate device, predicts likely verification outcomes while verification runs, and pre-speculates for them in a cache; a hit removes drafting latency. It is lossless in principle, but it consumes an extra device, cache-hit rate falls at larger batch or higher temperature, and it remains a preprint rather than a verified deployment. The paper itself notes that speculative decoding is generally ineffective for throughput-bound workloads such as large-scale RL rollouts. The 2026 claim should therefore be treated as an interesting prototype, not independent confirmation.


# Alternative Models and Emerging Workloads

**MoE** reduces FLOPs per token, not the memory required to hold all experts. At batch-one decode, active-expert weights and KV cache dominate bandwidth; at large expert parallelism, all-to-all communication becomes the bottleneck. DeepSeek-V2 and V3 make this concrete: fine-grained routed experts plus shared experts, device- or node-limited routing, auxiliary or bias-based balancing, and prefill/decode disaggregation. Their throughput figures are vendor-reported and tied to specific clusters.

**MLA** compresses KV by caching a low-rank latent plus a decoupled RoPE key. It changes the KV layout and therefore which quantization schemes transfer. **DeepSeek Sparse Attention** reduces main-attention compute by selecting a fixed top-k of KV entries, but the indexer itself remains quadratic and the full KV cache is still stored. Sparse attention is not a KV-memory solution.

**Mamba-2/SSD** shows the duality between structured state-space models and masked attention. Its state is constant-size, so decode cost does not grow with context, but that state is lossy for exact retrieval and copying. **Kimi Linear** is the strongest reviewed evidence that a hybrid can match full attention under a matched recipe: it interleaves a gated delta-rule linear layer with MLA at a 3:1 ratio and keeps a minority of full-attention layers. It still relies on those full-attention layers for retrieval, and fixed-state linear layers remain weaker at exact recall.

**Diffusion language models** trade sequential decode for parallel refinement. LLaDA is bidirectional and therefore incompatible with causal KV caching; it is competitive on some benchmarks but weaker on others, sensitive to sampling hyperparameters, and needs semi-autoregressive remasking for instruct behavior. Mercury commercializes diffusion for code and reports strong single-shot throughput, but Mercury 2 is a vendor blog/press claim with no technical paper or open weights. Diffusion is a better fit for single-shot, latency-sensitive generation than for multi-turn agents unless block-causal or hybrid caching is added.

**Test-time compute.** s1 shows that a small curated reasoning set plus “budget forcing”—suppressing the end-of-thinking token and appending “Wait,” or forcing termination—can induce test-time scaling cheaply. The effect flattens with extended budgets and is bounded by context length; rejection sampling can even scale inversely because longer samples are often the ones that went wrong. Reasoning workloads are therefore decode-heavy and long-output, which favors KV efficiency and low-batch exact speculation but works against speculation in throughput-bound RL rollouts.

**Agents.** The 2026 agent characterization is the most directly relevant systems evidence, but it is a preprint using two open models and one scaffold. Its structural findings are stronger than its exact percentages: turn counts are long-tailed; SWE-style tasks accumulate large contexts; with an ideal cache, most input tokens are prefix hits; decode dominates LLM time; and failed runs carry larger contexts. The authors stress that this is a no-thrashing regime. Once aggregate agent context exceeds GPU memory, eviction turns a decode-dominated workload back into expensive recomputation. Capacity planning should target resident KV bytes and eviction rate, not average prompt length.

**CacheBlend** extends reuse beyond the exact prefix by reusing precomputed chunk KV caches and selectively recomputing a small fraction of tokens to restore cross-chunk attention. It is transformer-specific and evaluated on RAG/QA chunks rather than agent traces, but the mechanism is directly relevant to persistent agent state.

**Multimodal.** NVILA’s “scale-then-compress” recipe increases spatial/temporal resolution and then compresses visual tokens through spatial-to-channel reshaping and temporal averaging. It reduces visual-token load and training cost, but accuracy is competitive rather than uniformly better, and visual-token KV behavior differs from text-only prefix reuse. Its efficiency claims are hardware- and configuration-specific.


# Existing Engines and the Current Research Landscape

| Engine (status) | Core design | Genuine strengths | Constraints / failure modes |
|---|---|---|---|
| **vLLM V1** (released; V0 deprecated) | PagedAttention, unified token-budget scheduler, chunked prefill and prefix caching default, TP, piecewise CUDA graphs | Broad model/hardware coverage, strong continuous batching, low CPU overhead, easy OpenAI-compatible serving | Disaggregation depends on connector/backend; raw throughput and goodput differ; default preemption uses recomputation, with explicit offloading connectors available; prefix locality depends on routing; CPU overhead still matters for small models [16](#ref-systems-16)[17](#ref-systems-17)[18](#ref-systems-18) |
| **SGLang** (released) | RadixAttention, cache-aware scheduling, HiCache L1/L2/L3, PD/EPD disaggregation, structured output, large-scale EP | Strong prefix reuse for multi-turn/agentic/RAG; no universal ranking established; hierarchical cache; DeepSeek-scale MoE | Greedy cache-aware scheduling can starve; L2 is instance-private; L3 correctness/latency depends on backend; PD adds transfer, timeout, and role-configuration complexity [8](#ref-systems-8)[19](#ref-systems-19) |
| **TensorRT-LLM** (released) | TensorRT compiled engines and a PyTorch-native backend, in-flight batching, paged KV, chunked context, FP8/FP4, XQA, NIXL disaggregation | Deep NVIDIA kernel/quantization control; strong TTFT/TPOT tuning; heterogeneous TP/PP per phase | NVIDIA-only; build/tuning burden; packed-input ordering; disaggregation is decoder-only with homogeneous per-layer KV; mixed context/generation scheduling is not optimal; unset `cache_transceiver_config` silently disables disagg, and transfer-setting mismatches fail at runtime [20](#ref-systems-20)[21](#ref-systems-21) |
| **llama.cpp** (released) | GGUF, CPU/GPU offload, continuous batching default, unified KV buffer, slots, prompt cache/checkpoints, speculative decoding | Edge/desktop and hybrid CPU+GPU; broad quantization; simple deployment | Primarily single-host, with an optional remote-device RPC backend; no general production cluster scheduler or PD stack; slot-based parallelism; unified KV and context shift have different failure modes; throughput ceiling under high concurrency [22](#ref-systems-22) |
| **ONNX Runtime GenAI** (preview API) | Dynamic batching, paged KV, prefix caching, transactional engine, per-turn cancellation, hybrid paged/fixed state | Portable ONNX execution; explicit cancellation/rollback; works across execution providers | Single-owner thread; exactly one paged-KV group; beam search rejected; static fallback lacks the transactional path; prefix hits require full blocks plus parent identity; not a cluster scheduler [23](#ref-systems-23) |

**Orchestration is not inference.** NVIDIA Dynamo (March 2025) adds a Planner, KV-aware Smart Router, distributed KV-cache manager, and NIXL transport above TensorRT-LLM/vLLM/SGLang/PyTorch; llm-d (May 2025) is the Kubernetes-native equivalent with vLLM and Inference Gateway. They solve placement, routing, tiering, and autoscaling, but their gains depend on connector maturity, network, cache backend, routing quality, and “projected performance” vendor caveats [24](#ref-systems-24)[25](#ref-systems-25).

Released and documented: vLLM V1 (V0 deprecated; chunked prefill and prefix caching default; NIXL, Mooncake, LMCache, offloading connectors), SGLang HiCache and PD, TensorRT-LLM IFB and NIXL disaggregation, llama.cpp continuous batching, ONNX RT GenAI dynamic batching, Dynamo, and llm-d. Research prototypes: Sarathi-Serve (fork), DistServe, Splitwise, and AMPD. Mooncake is production at Kimi but its full stack is not a drop-in OSS engine. FA4 is the 2026 Blackwell kernel paper; FA3 is the Hopper reference. I found no independently verified 2026 evidence that one engine dominates all workloads.

| Bottleneck | Key works | Contribution | Distinction or limit |
|---|---|---|---|
| Attention IO | [FlashAttention](https://arxiv.org/abs/2205.14135), [FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451), [FlashInfer](https://arxiv.org/abs/2501.01005) | Tiling, online softmax, work partitioning, paged/ragged attention | Kernel results are not serving results; hardware, head dimension, and shape matter |
| Static batching | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level scheduling and selective batching | Needs memory admission and fairness, not just a queue |
| KV fragmentation and reuse | [PagedAttention](https://arxiv.org/abs/2309.06180), [SGLang](https://arxiv.org/abs/2312.07104) | Block tables, copy-on-write, radix-tree prefix reuse | Paging adds indirection; reuse requires compatible model state and exact prefixes |
| Prefill/decode interference | [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677), [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) | Chunked prefill, disaggregated pools, cache-aware placement | Transfer, queueing, and orchestration can erase gains; colocated remains the default |
| Weight/state bandwidth | [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438), [FP8](https://arxiv.org/abs/2209.05433), [NVFP4](https://arxiv.org/abs/2509.25149), [KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079) | Fewer bytes per weight, activation, or KV element | All are approximate; raw bit width is not realized memory or realized latency |
| Sequential decode | [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html), [Chen et al.](https://arxiv.org/abs/2302.01318), [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE-3](https://arxiv.org/abs/2503.01840), [SSD](https://arxiv.org/abs/2603.03251) | Draft/verify, tree attention, asynchronous drafting | Exactness contracts differ; gains collapse when verification becomes compute-bound |
| Model state | [MLA](https://arxiv.org/abs/2405.04434), [Mamba-2](https://arxiv.org/abs/2405.21060), [Kimi Linear](https://arxiv.org/abs/2510.26692), [LLaDA](https://arxiv.org/abs/2502.09992) | Compression, constant state, hybrid layers, diffusion | Checkpoint-specific; retrieval, caching, and quality tradeoffs differ |

Keep these distinctions explicit: attention-kernel throughput versus end-to-end serving; prefill versus decode; capacity versus bandwidth; exact distribution-preserving speculation versus approximate acceptance; weight-only versus activation/KV quantization; prefix reuse versus non-prefix reuse; colocated versus disaggregated serving. Published speedups are hypotheses to test on your workload, not multiplicative factors.

**Exactness.** Exact speculative sampling preserves the target distribution under its assumptions and numerical implementation; it does **not** guarantee the same sampled sequence for the same seed. Greedy verification is a different contract. Approximate Speculative Decoding (ASD) is a **greedy** verifier that replaces first-mismatch truncation with a budgeted longest-prefix rule using target-logit regret, a per-block exception cap, and a request-level budget; it explicitly does not guarantee identical outputs, semantic preservation, or task correctness ([ASD, arXiv 2608.03447](https://arxiv.org/abs/2608.03447)). It is not a drop-in relaxation of stochastic rejection sampling. The paper reports author-run experiments on one L20 with vLLM 0.11.0 and no independent reproduction.

**Adaptive verification.** The vLLM estimator uses an online per-position logistic regression and an IRLS refit, with a token-independent \(\operatorname{logit}(q_{\max})\) feature. Trimming on \(q_{\text{sampled}}\) was rejected because the trim decision would correlate with the sampled token and break the proposal distribution. This was a design constraint, not an unresolved risk. Adaptive verification changes the number of drafted tokens **verified**, not necessarily the number drafted, and requires variable-length verification support; gains are smaller for auto-regressive drafters that pay to draft tokens later trimmed ([14 October 2026](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator)).

**Hybrid state.** A hybrid model’s cache is not one KV tensor. Full-attention layers retain KV; linear/recurrent layers retain a fixed-size state; some designs add replay buffers or checkpoints. Prefix identity must include the recurrent checkpoint, not just token IDs. A finite recurrent state compresses history; hybrid designs retain some full-attention layers. Replay can be exact or approximate depending on the algorithm ([Kimi Linear](https://arxiv.org/abs/2510.26692), [Marconi](https://arxiv.org/abs/2411.19379)). [ReplaySSM](https://tridao.me/blog/2026/replayssm/) is author-described exact buffered replay up to floating-point error; [SuffixReplay](https://arxiv.org/abs/2609.33477) is an abstract-only approximate reconstruction proposal.

**Quantization.** GPTQ/AWQ are weight-only: they reduce bytes moved, not arithmetic, so gains shrink as batch makes decode compute-bound. SmoothQuant is an 8-bit W8A8 method with outlier migration. FP8/NVFP4 are hardware- and kernel-dependent; NVFP4 is Blackwell-oriented, and small or sensitive models may need quantization-aware distillation. KV methods such as KIVI and KVQuant assume MHA-like key-outlier structure and specific RoPE placement; MLA’s latent/decoupled key, GQA/MQA’s fewer heads, and hybrid layers change both the savings and the method. Exact speculation does not undo quantization error, because the “target” being verified is the quantized model.

Dates and content were checked; results are author-reported unless noted.

| Date | Primary source | Evidence status and engineering implication |
|---|---|---|
| 5 Mar 2026 | [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) | Author-reported Blackwell kernel co-design; code released; not verified as every engine’s default. Appendix platform description differs from the headline B200 claim. |
| 25 Apr 2026 | [SGLang DeepSeek-V4 support](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) | Author-reported hybrid sparse attention, compression, ShadowRadix, heterogeneous KV/state pools; 1M-context claim. Requires model-specific state semantics. |
| May 2026 | [Agentic workload characterization](https://arxiv.org/abs/2605.26297) | Preprint; two open models and one scaffold. Under an ideal cache, most input tokens are prefix hits and decode dominates LLM time; once aggregate context exceeds memory, eviction restores expensive recomputation. |
| 15 Jun / 27 Sep 2026 | [ReplaySSM](https://tridao.me/blog/2026/replayssm/), [SuffixReplay](https://arxiv.org/abs/2609.33477) | ReplaySSM author-described; SuffixReplay abstract-only. Both explore reconstructing recurrent state rather than retaining exact history. |
| 29 Sep 2026 | [vLLM disaggregation guide](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide) | Documented operational boundary: frontend/render, KV connectors, router, workers. The guide shows slow transfer can erase P/D gains. |
| 12 Oct 2026 | [Bidirectional KV transfer](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload) | Author-reported. Decode retains **prompt and generated KV** (decode-retained state), with block metadata returned to prefill; TTL and recompute thresholds are configuration-sensitive. |
| 14 Oct 2026 | [Push-based NIXL connector](https://vllm.ai/blog/2026-10-14-nixl-push-connector) | Author-reported. Push moves setup off the first-token critical path but commits decode placement earlier; pull retains load-aware routing and is preferred for bidirectional transfer. |
| 14 Oct 2026 | [Adaptive verification estimator](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator) | Author-reported. Online estimator, \(\operatorname{logit}(q_{\max})\), variable-length verification; smaller gains for MTP/EAGLE drafters. |

Released/documented serving capabilities include vLLM V1’s unified token-budget scheduler, chunked prefill, prefix caching, and disaggregation connectors; SGLang HiCache and PD disaggregation; TensorRT-LLM in-flight batching, paged KV, and NIXL disaggregation; llama.cpp continuous batching with slot-based KV; ONNX Runtime GenAI dynamic batching and paged KV; and Dynamo/llm-d orchestration. Sarathi-Serve, DistServe, Splitwise, AMPD, SSD, ASD, and SuffixReplay remain research prototypes or preprints. FA4 has released code but is not verified as a default backend in the major engines.

The recovery agent edited the source agent's research; first-person reading claims refer to that original run. Benchmark figures above are author-reported historical observations, and the memo does not reproduce every workload/configuration detail needed for comparison. Do not rank engines from those figures. See [final review](../review_memos.md) for missing conditions.

Decode context parallelism is documented in [vLLM DCP](https://vllm.ai/blog/2026-08-07-decode-context-parallelism); CP is not limited to prefill. TensorRT-LLM has a [PyTorch-native backend](https://nvidia.github.io/TensorRT-LLM/torch.html), so compiled-engine constraints do not describe every path. llama.cpp exposes remote devices through its [RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md). Explicit vLLM CPU offload connectors are distinct from default recompute-based scheduler preemption. MLA replication depends on the sharding layout. Mooncake's link-rate recommendation is deployment-specific, not a universal disaggregation threshold.


# Designing and Building an Inference Engine

**Recommendation, not a literature finding.** Build a small, numerically trustworthy engine first, then make state and scheduling first-class.

```
client → tokenizer/render → admission & scheduler → execution plan
       → model/state contracts → kernel backend → logits/sampling → stream/cancel
state allocator ↔ prefix cache ↔ optional transfer/offload
metrics & correctness checks span scheduler, state, and kernels
```

Make these interfaces explicit: tensor dtype/shape/stride/layout; weight packing and quant format; per-layer state descriptor with alloc, append, fork, truncate, checkpoint, and release; scheduler token and memory budgets; asynchronous completion; sampling and RNG ownership; backend capability discovery with explicit rejection of unsupported combinations. Cache identity must include checkpoint revision, adapter, token IDs, positional configuration, and relevant multimodal inputs. Shared state must be immutable or copy-on-write; cancellation and speculative rejection must release or roll back exactly the state they own.

| Stage | Deliverable | Exit criterion |
|---|---|---|
| 0 | Workload specification: checkpoint, hardware, context/output distribution, concurrency, quality tolerance, SLO | Written benchmark spec and baseline |
| 1 | Correct reference decoder, tokenizer, loader, full forward, sampling | Logits/activations match a trusted implementation within stated tolerance |
| 2 | Cached single request with prefill/decode split | Cached logits match full recomputation across boundary cases |
| 3 | Efficient single device: vendor GEMM/attention, fused ops, workspace reuse | Profile explains time and memory; measured end-to-end gain |
| 4 | Concurrent serving: continuous batching, allocator, chunked prefill, cancellation | Mixed-length stress tests; stable memory; useful SLO-goodput |
| 5 | Prefix reuse and one validated quantization path | Isolation/eviction tests and quality evaluation pass |
| 6 | Optional speculation, parallelism, or disaggregation | Benefit survives realistic load, transfer cost, and quality constraints |

Start colocated with chunked prefill and prefix caching. Move to disaggregation only when TTFT and TPOT SLOs demonstrably conflict and KV transfer is fast relative to prefill. Choose kernels by phase: prefill/long-context kernels for prompt work, paged decode and split-K workhorses for token generation. Use TP inside an NVLink domain, PP across nodes, EP for MoE with load balancing, and CP mainly for long prefill—not for sharding decode KV without accounting for per-token communication. Invest in the KV hierarchy before exotic scheduling.

A swappable serving stack needs explicit contracts:

1. **Request:** OpenAI-compatible chat/completions plus token-in/token-out; streaming; request ID stable across prefill and decode; cancellation; priority/deadline or SLO class; cache hint; tenant and idempotency keys.
2. **Scheduler:** token budget (`max_num_tokens`/`max_scheduled_tokens`), max batch size, chunk size, prefill/decode pool selection, admission/rejection (429 with retry-after), FCFS versus priority, preemption (recompute vs swap), and KV reservation policy (SGLang’s `prefill_complete` versus `early`).
3. **KV/cache:** page/block size, block table, prefix hash plus parent-block identity (hash alone is insufficient), tier L1/L2/L3, `put/get/query/evict`, eviction, replication, TTL, tenant isolation, and MLA versus MHA layout.
4. **Transfer:** connector abstraction (NIXL, Mooncake, LMCache, custom), layout transformation for heterogeneous TP/PP, timeout, retry, backpressure, idempotent transfer IDs, and asynchronous overlap with compute.
5. **Execution:** attention backend (FA2/FA3/FA4/FlashInfer/XQA), CUDA graphs or piecewise capture, compile mode, quantized KV, speculative decoding, structured output.
6. **Observability and failure:** per-request queue, prefill, decode, TTFT/TPOT/ITL histograms, cache-hit tokens, transfer time, preemption count, SLO attainment and goodput, joined by request ID. Define cancellation, timeout, partial-KV, duplicate-ID, and whether the engine rolls back or fails stop; ONNX’s dynamic path is transactional before publication and fail-stop after [23](#ref-systems-23).

1. A reference decoder-only forward pass with a tokenizer and logits parity against an existing implementation on short prompts. Correctness before performance.
2. A contiguous per-sequence KV cache for MHA, with prefill and decode paths. Test cache-vs-full-recompute logits at multiple lengths and positions.
3. GQA/MQA support by caching only g KV heads and mapping query heads to KV groups. Do not materialize h copies.
4. A naive attention kernel first; then a tiled/IO-aware kernel or a vendor attention call. Measure achieved bandwidth before and after.
5. A small graph/IR with shape specialization, op fusion, and a memory planner. Even a simple engine benefits from separating graph optimization from kernel selection.
6. Roofline instrumentation: count FLOPs and bytes per phase, compute operational intensity, and report achieved bandwidth and MFU. This tells you whether to optimize compute or memory.
7. Weight-only quantization for decode; KV-cache quantization later, with accuracy checks. Quantization changes b and adds scale metadata, so re-derive the cache formula.
8. Batching and continuous batching only after single-sequence prefill/decode is correct and profiled. That is a separate serving problem.
9. MLA only if the model requires it: implement the absorbed projections and decoupled RoPE correctly, and cache c_KV plus k_R, not per-head K/V.
10. For state-space models, implement a recurrent-state cache and a scan kernel; do not force them into a KV-cache abstraction.

What not to do first: distributed multi-chip serving, speculative decoding, or custom GEMMs. The dominant early wins are correctness, cache layout, and memory traffic.

1. **Start colocated** with chunked prefill and prefix caching; measure SLO attainment before paying for disaggregation.
2. **Disaggregate only on evidence:** conflicting TTFT/TPOT SLOs, long prompts or high reuse, and measured KV-transfer cost small enough for the target SLO. Report raw throughput and SLO-constrained goodput separately; no universal link-rate threshold applies.
3. **Pick kernels by phase and hardware:** FA3/FA4 for prefill/long context on Hopper/Blackwell; FlashInfer/FA2-class paged/varlen kernels or XQA for decode. Pin versions and shapes.
4. **Invest in the cache before the split:** local radix/prefix cache, KV-aware routing, then a measured L3. Enforce tenant isolation and block-identity correctness.
5. **Map parallelism to topology:** TP inside NVLink, PP across nodes, EP with load balancing for MoE, CP for long prefill; decode CP when its KV-capacity benefit outweighs per-token communication, DP plus KV-aware routing for replicas.
6. **Report SLO-goodput at p95/p99 under open-loop traffic with pinned configurations.** A throughput number without SLOs, hardware, workload, and comparator is not comparable.


# Measurement, Correctness, and Reproducibility

**Metrics.** TTFT is arrival to first generated token; state whether tokenization, preprocessing, queueing, and network are included. TPOT is `(last_token_time − first_token_time)/(output_tokens − 1)`, but report the full ITL/TBT distribution because TPOT hides stalls. Report p50/p95/p99 with sample size and estimator (nearest-rank versus interpolated); p99 on a few hundred requests is noisy. SLO-goodput is the maximum rate meeting both TTFT and TPOT at a stated attainment (e.g., 90% or 99%), per GPU; publish the full attainment curve, not one point. Throughput/cost means output tokens/s, requests/s, and cost per 1M output tokens; report energy only if measured.

**Traffic and matrix.** Use open-loop arrivals independent of server latency. Poisson is a weak default; include bursty/Markov-modulated or real traces and multi-turn/agentic interleaved prefills. Cover ISL/OSL distributions, prefix-reuse rate, long context, dense versus MoE, reasoning/long output, BF16/FP8/FP4, TP/PP/EP/DP, block size, chunk size, `max_num_tokens`, KV budget, cache on/off, and SLO targets. Compare colocated, chunked-prefill, and disaggregated configurations at the same GPU count and attainment.

**Reproducibility and ablations.** Pin engine version/commit, CUDA/driver, GPU SKU and clocks/power, model/tokenizer revision, quantization, parallelism, block size, batch-token cap, chunk size, scheduling policy, cache state, seed, trace hash, warmup, run count, and percentile method; publish raw latency distributions. Keep kernel microbenchmarks (TFLOP/s, bandwidth, occupancy) separate from end-to-end serving. Ablate colocated versus disaggregated, chunked prefill versus disagg, prefix caching on/off, TP versus PP, EP size and redundant experts, attention backend, and scheduler policy. Inject KV-transfer timeout/corruption, prefill or decode crash, L3 outage, overload/429, cancellation storms, and duplicate IDs; measure recovery time and SLO attainment during failure. Note that greedy outputs can change across batch compositions because kernels reassociate floating point, and that prompt-cache reuse can change logits [22](#ref-systems-22)[23](#ref-systems-23).

**Recommendation.** Define metrics before measuring. TTFT is request arrival to first streamed token; state whether tokenization, preprocessing, queueing, and network are included. TPOT is \((t_{\text{last}}-t_{\text{first}})/(N_{\text{out}}-1)\), but also report the full inter-token-latency distribution. SLO-goodput is the maximum request rate meeting both TTFT and TPOT/ITL targets at a stated attainment level, per GPU. Report p50/p95/p99, errors, rejected requests, offered versus achieved load, output tokens/s, requests/s, cost per million tokens, and energy per token where relevant.

Use open-loop arrivals—a fixed timestamp trace or bursty stochastic process—not a closed-loop client that throttles on latency. Poisson is a weak default for production; include burstiness, multi-turn/agentic interleaving, and prefix-reuse skew. Warm up, measure steady state, and separate queueing from execution. Cross input/output lengths, shared and independent prefixes, cold/warm caches, low-to-saturating concurrency, several context lengths, and cancellation. Compare disaggregation at the **same GPU count and SLO attainment**, not against a colocated system given more resources.

Pin engine commit, CUDA/driver, GPU SKU, clocks and power limits, model/tokenizer revision, quantization, parallelism, block size, max batch tokens, chunk size, scheduling policy, cache state, seed, trace hash, warmup, and run count. Publish raw latency samples and keep kernel microbenchmarks separate from end-to-end serving. Include ablations and failure injection: transfer timeout, prefill/decode crash, cache-backend outage, overload/429, and cancellation storms. Correctness tests should cover causal masks, padding/ragged batches, GQA head mapping, positional offsets, cache append/fork/truncate, block boundaries, EOS/stop behavior, all-accepted and all-rejected speculation, numerical stability, and quantization quality. Stochastic sampling needs distribution-oriented tests, not only fixed-seed string equality.


# Research Directions and Falsifiable Hypotheses

These are **predictions/proposals**, not established findings.

| Hypothesis | Controlled experiment | Success metric | What would falsify it |
|---|---|---|---|
| Joint scheduling of prefill chunks and speculative verification beats fixed independent settings | Fixed vs adaptive policies on the same mixed trace | Higher SLO-goodput at equal quality and hardware | Controller overhead or instability outweighs savings |
| A typed per-layer state interface supports attention, compressed, and recurrent layers without material abstraction cost | Same model/kernels behind specialized vs common state APIs | Equivalent numerics and small measured overhead; reuse across architectures | The abstraction prevents critical fusion/layout choices |
| Reuse-and-transfer-cost-aware cache placement beats recency/LRU | Replay multi-turn agent traces at fixed capacity and bandwidth | Lower p95 TTFT and recompute FLOPs at stable hit rate | Prediction errors cause churn and evict useful prefixes |
| Quality-budgeted per-component precision improves task-success per joule | Sweep weight/activation/KV precision at fixed task-quality threshold | Lower energy and latency at matched task success | Error accumulates on long context or reasoning, or QAD erases the gain |
| Session-aware routing reduces agent completion cost versus load-only routing | Same multi-step tasks under locality-aware vs load-only routing | Lower cost and completion latency at equal success | Locality concentrates load and increases queueing |

**H1 (state over FLOPs).** For long-horizon agent traces with high prefix-cache hit rates and no KV thrashing, an intervention that reduces attention FLOPs but leaves resident KV bytes unchanged will produce a smaller end-to-end latency improvement than one that reduces resident KV bytes by a comparable factor. Falsified if compute-only attention sparsification consistently wins across models and traces.

**H2 (speculation ceiling).** Exact speculative decoding is a latency optimization whose benefit shrinks as batch size and target compute intensity rise; increasing draft acceptance cannot compensate once verification becomes compute-bound. Falsified by a workload where higher acceptance yields monotonic throughput gains at large batch.

**H3 (KV transfer).** KV-quantization quality loss is predicted by mismatch between the method’s outlier assumptions and the attention layout: per-channel key methods will degrade more on MLA/MQA than on MHA at matched bit-width, and removing the recent-token residual window will hurt multi-step reasoning more than open-ended chat. Falsified if degradation is uniform across attention variants and residual-window ablations.

**H4 (hybrid retrieval).** A hybrid linear/full-attention model with a minority of full-attention layers will match full attention on retrieval-heavy long-context benchmarks, while a pure linear/SSM model will fail exact retrieval as context and retrieval density grow. Falsified if a pure linear/SSM model matches full attention on RULER-style exact retrieval.

**H5 (diffusion/agent mismatch).** Diffusion LMs’ single-shot throughput advantage will not transfer to multi-turn agent workloads with high prefix reuse unless block-causal or hybrid caching is added; per-turn latency will exceed an autoregressive engine with prefix caching. Falsified if a bidirectional diffusion LM without KV caching beats an autoregressive engine on agent traces.

1. **Foundations:** McCulloch–Pitts (1943), Rosenblatt (1958), Rumelhart et al. (1986), Shannon (1951), Bengio et al. (2003), LSTM (1997), seq2seq (2014), Bahdanau et al. (2015), Transformer (2017), GPT-1 and GPT-3.
2. **Mechanics:** Pope et al. on prefill/decode ([2022](https://arxiv.org/abs/2211.05102)), the roofline paper, MQA → GQA → MLA, FlashAttention, then Mamba.
3. **Serving systems:** Orca, PagedAttention/vLLM, SGLang, Sarathi-Serve, DistServe/Splitwise/Mooncake, FlashInfer, and the vLLM V1 and HiCache documentation.
4. **Algorithms:** GPTQ/AWQ/SmoothQuant, FP8/NVFP4, KIVI/KVQuant, then exact speculative decoding (Leviathan/Chen), Medusa, EAGLE-3, SSD, and ASD.
5. **Workloads:** the agent-workload characterization, CacheBlend, Kimi Linear, LLaDA/Mercury, and s1.
6. **2026 sources:** FA4, SGLang DeepSeek-V4, the vLLM disaggregation/bidirectional/push/adaptive-verification posts, ReplaySSM, SuffixReplay, and AMPD—then re-derive the cost model against your own hardware.

**Tier 1 — mechanics before writing a kernel:** Vaswani et al. 2017; Pope et al. 2023; Williams et al. 2009; Shazeer 2019; Ainslie et al. 2023; DeepSeek-AI 2024; Dao et al. 2022.

**Tier 2 — runtime and compilers:** Chen et al. 2018 (TVM); OpenXLA architecture docs; ONNX Runtime architecture docs; Kaplan et al. 2020.

**Tier 3 — historical grounding:** McCulloch and Pitts 1943; Rosenblatt 1958; Rumelhart et al. 1986; Shannon 1951; Bengio et al. 2003; Hochreiter and Schmidhuber 1997; Sutskever et al. 2014; Bahdanau et al. 2015; Radford et al. 2018 (GPT-1); Brown et al. 2020 (GPT-3).

**Tier 4 — the alternative state model:** Gu and Dao 2023 (Mamba).


# Study Exercises {.unnumbered}

1. **GEMM arithmetic.** A dense GEMM costs approximately $2MKN$ FLOPs when one multiply-add counts as two FLOPs. For $M=K=N=4096$, compute the FLOPs and the time floor at $R_{\text{peak}}=10^{14}$ FLOP/s. State what the floor ignores.

2. **Dense parameter formula.** For a decoder with $L$ layers, width $d$, vocabulary $V$, MHA, a non-gated two-matrix MLP of width $4d$, tied embeddings, and no biases, the report uses $P\approx 12Ld^2+Vd$. Compute $P$ for $L=32$, $d=4096$, $V=32{,}000$. Name two architecture changes that break the formula. For an “8B” dense checkpoint in BF16, compute raw weight bytes in GB and GiB.

3. **KV cache example.** Use $M_{\text{GQA}}=2LBngd_hb$ with $L=32$, $B=1$, $n=8192$, $g=8$, $d_h=128$, and $b=2$ bytes. Compute KV bytes per request and for 16 equal requests, in GiB and decimal GB.

4. **Batch weight traffic versus FLOPs.** Let $P_w$ be active linear parameters, $b_w$ bytes per weight, batch $B$, peak $R_{\text{peak}}$, and HBM bandwidth $BW$. Assume each decode step reads all weights once and performs $2P_wB$ linear FLOPs. Derive the linear arithmetic intensity and the critical batch where it reaches the ridge point. Evaluate for $b_w=2$ B, $R_{\text{peak}}=10^{14}$ FLOP/s, and $BW=3.35\times10^{12}$ B/s. Explain the implication for weight-only quantization as $B$ grows.

5. **Optimistic roofline floor.** For one decode step assume $F=1.28\times10^{11}$ FLOP, $D_{\text{HBM}}=1.6\times10^{10}$ B, $D_{\text{net}}=8\times10^{9}$ B, $R_{\text{peak}}=10^{14}$ FLOP/s, $BW_{\text{HBM}}=3.35\times10^{12}$ B/s, and $BW_{\text{net}}=10^{11}$ B/s. Compute the floor
\[
t\ge\max\left(\frac{F}{R_{\text{peak}}},\frac{D_{\text{HBM}}}{BW_{\text{HBM}}},\frac{D_{\text{net}}}{BW_{\text{net}}}\right).
\]
Why is adding the three times incorrect?

6. **Paging overhead.** Blocks hold 32 tokens. Two requests have 65 and 31 useful tokens. Compute blocks and slots allocated to each, total allocated and useful slots, unused slots, unused fraction of allocated capacity, and unused slots relative to useful slots. Explain why the last two percentages differ.

7. **Speculative exactness.** Define the exactness contract of distribution-preserving speculative sampling. Why does it not imply identical output strings for the same seed? What changes when the target model is quantized?

8. **Throughput versus SLO goodput.** Design an open-loop experiment that separates maximum throughput from SLO-goodput. Specify the sweep variable, the SLOs, and the measurements that reveal the difference.

# Worked Answers {.unnumbered}

1. $2MKN=2(4096)^3=1.374\times10^{11}$ FLOP. With $R_{\text{peak}}=10^{14}$ FLOP/s, $t\ge F/R_{\text{peak}}=1.374\times10^{-3}$ s, or 1.37 ms. This assumes a dense operation with no memory, launch, dependency, or scheduling cost; it is an idealized floor, not a runtime prediction.

2. $12Ld^2=12\cdot32\cdot4096^2=6.442\times10^9$ parameters; $Vd=32{,}000\cdot4096=1.311\times10^8$; so $P\approx6.57\times10^9$. GQA/MQA change head and KV dimensions; gated MLPs add projections; MoE activates only some experts; biases, untied embeddings, and vocabulary padding also break the formula. For 8 billion BF16 weights:
\[
8\times10^9\times2=1.6\times10^{10}\ \text{B}=16\ \text{GB}=14.90\ \text{GiB}.
\]
Realized memory is higher after scales, metadata, allocator slack, and workspaces.

3. $M=2\cdot32\cdot1\cdot8192\cdot8\cdot128\cdot2=1{,}073{,}741{,}824$ B. That is $2^{30}$ B $=1$ GiB $=1.074$ GB. Sixteen requests: $1.718\times10^{10}$ B $=16$ GiB $=17.18$ GB. Assumes equal lengths, no fragmentation, no metadata, and BF16 KV.

4. $D_w\approx P_wb_w$ bytes per step, independent of $B$ under the one-weight-read assumption. $F\approx2P_wB$, so $I=F/D_w=2B/b_w$ FLOP/B. The ridge is $I_{\text{ridge}}=R_{\text{peak}}/BW=10^{14}/(3.35\times10^{12})=29.9$ FLOP/B. Setting $2B/b_w=I_{\text{ridge}}$ gives $B_{\text{crit}}=b_wR_{\text{peak}}/(2BW)\approx29.9$. At $B=1$, $I=1$ FLOP/B, below the ridge; at $B=64$, $I=64$, above it. Weight-only quantization lowers $b_w$, saving bandwidth at small $B$ but shifting $B_{\text{crit}}$ downward; gains shrink once compute or KV traffic dominates.

5. Compute: $1.28\times10^{11}/10^{14}=1.28\times10^{-3}$ s. HBM: $1.6\times10^{10}/(3.35\times10^{12})=4.78\times10^{-3}$ s. Network: $8\times10^9/10^{11}=8.0\times10^{-2}$ s. The floor is $8.0\times10^{-2}$ s, or 80 ms, set by network. Adding gives about 86.1 ms, which assumes no overlap and is not the model’s bound; the roofline floor assumes perfect overlap. Real time can exceed 80 ms because of dependencies, synchronization, queueing, and other overheads.

6. Request A: $\lceil65/32\rceil=3$ blocks $=96$ slots. Request B: $\lceil31/32\rceil=1$ block $=32$ slots. Allocated $=128$ slots; useful $=96$; unused $=32$. Unused fraction of allocated capacity $=32/128=25\%$. Unused relative to useful $=32/96=33.3\%$. The first answers “what fraction of purchased slots is idle?”; the second answers “how much extra capacity is paid per useful token?”. Per request, A wastes $31/96=32.3\%$ of its allocation and B wastes $1/32=3.1\%$.

7. Exact speculative sampling accepts a draft token with probability $\min(1,p/q)$ and, on rejection, samples from the normalized residual $\max(0,p-q)$, so output is distributed as target $p$ under correct $p,q$ and arithmetic. Same-seed strings are not guaranteed because draft and target consume different random draws and kernel reduction orders; the distribution, not the realized sequence, is preserved. With a quantized target, “exact” means exact for $p_q$; quantization error relative to the full-precision model remains.

8. Fix model, GPU count, input/output length distributions, cache state, and arrival trace. Sweep offered request rate with open-loop arrivals, not a closed-loop client. Define SLOs such as p95 TTFT $\le2$ s and p95 TPOT $\le50$ ms. Measure achieved requests/s, output tokens/s, p50/p95/p99 TTFT and TPOT, rejects/errors, and queue delay. Maximum throughput is the highest achieved rate regardless of latency; SLO-goodput is the highest rate meeting both latency targets at the stated attainment. Throughput can rise while goodput saturates or falls as queues grow.



# References and Source Access {.unnumbered}

The following catalogs retain the original source agents' access notes. Duplicate works may appear in more than one research track because the inspected material and role differed. These notes describe reading, not independent replication.



## Foundations references

| Title | Authors / organization | First publication date | URL | What was actually read |
|---|---|---|---|---|
| A Logical Calculus of the Ideas Immanent in Nervous Activity | Warren S. McCulloch; Walter Pitts | 1943-12 (Bull. Math. Biophys. 5(4):115–133) | [link](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf) | Full PDF (19 pp.) |
| The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain | Frank Rosenblatt | 1958 (Psych. Rev. 65(6):386–408) | [link](https://psycnet.apa.org/record/1959-09865-001) | Metadata/abstract only; PDF fetches timed out |
| Learning Representations by Back-Propagating Errors | David E. Rumelhart; Geoffrey E. Hinton; Ronald J. Williams | 1986-10-09 (Nature 323:533–536) | [link](https://www.nature.com/articles/323533a0) | Publisher abstract; full text paywalled |
| Prediction and Entropy of Printed English | Claude E. Shannon | 1951-01 (BSTJ 30(1):50–64) | [link](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) | Full PDF (12 pp.) |
| A Neural Probabilistic Language Model | Yoshua Bengio; Réjean Ducharme; Pascal Vincent; Christian Jauvin | 2003-02 (JMLR 3:1137–1155) | [link](https://jmlr.org/papers/v3/bengio03a.html) | Abstract page; full PDF not read |
| Long Short-Term Memory | Sepp Hochreiter; Jürgen Schmidhuber | 1997 (Neural Computation 9(8):1735–1780) | [link](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf) | Full PDF (44 pp.) |
| Sequence to Sequence Learning with Neural Networks | Ilya Sutskever; Oriol Vinyals; Quoc V. Le | 2014-09-10 (arXiv; NeurIPS 2014) | [link](https://arxiv.org/abs/1409.3215) | arXiv metadata/abstract |
| Neural Machine Translation by Jointly Learning to Align and Translate | Dzmitry Bahdanau; Kyunghyun Cho; Yoshua Bengio | 2014-09-01 (arXiv; ICLR 2015) | [link](https://arxiv.org/abs/1409.0473) | arXiv metadata/abstract |
| Attention Is All You Need | Ashish Vaswani et al. | 2017-06-12 (arXiv; NeurIPS 2017) | [link](https://arxiv.org/abs/1706.03762) | arXiv metadata/abstract |
| Improving Language Understanding by Generative Pre-Training | Alec Radford; Karthik Narasimhan; Tim Salimans; Ilya Sutskever | 2018-06-11 (OpenAI preprint) | [link](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Full PDF (12 pp.) |
| Language Models are Few-Shot Learners | Tom B. Brown et al. | 2020-05-28 (arXiv; NeurIPS 2020) | [link](https://arxiv.org/abs/2005.14165) | arXiv metadata/abstract |
| Fast Transformer Decoding: One Write-Head is All You Need | Noam Shazeer | 2019-11-06 (arXiv) | [link](https://arxiv.org/abs/1911.02150) | arXiv metadata/abstract |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Joshua Ainslie et al. | 2023-05-22 (arXiv; EMNLP 2023) | [link](https://arxiv.org/abs/2305.13245) | arXiv metadata/abstract |
| DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2024-05-07 (arXiv) | [link](https://arxiv.org/abs/2405.04434) | Full HTML (architecture + appendix) |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao; Daniel Y. Fu; Stefano Ermon; Atri Rudra; Christopher Ré | 2022-05-27 (arXiv; NeurIPS 2022) | [link](https://arxiv.org/abs/2205.14135) | arXiv metadata/abstract |
| Roofline: An Insightful Visual Performance Model for Multicore Architectures | Samuel Williams; Andrew Waterman; David Patterson | 2009-04 (CACM 52(4):65–76) | [link](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) | Full PDF (10 pp.) |
| TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | Tianqi Chen et al. | 2018-02-12 (arXiv; OSDI 2018) | [link](https://arxiv.org/abs/1802.04799) | arXiv metadata/abstract |
| XLA: Accelerated Linear Algebra (architecture docs) | OpenXLA Project | Living doc; pages updated 2024-01-10 and 2024-12-03 | [link](https://openxla.org/xla/architecture) | Docs pages (full) |
| ONNX Runtime Architecture | Microsoft / ONNX Runtime project | Living doc; accessed 2026-10-19 | [link](https://onnxruntime.ai/docs/reference/high-level-design.html) | Docs page (full) |
| Efficiently Scaling Transformer Inference | Reiner Pope et al. | 2022-11-09 (arXiv; MLSys 2023) | [link](https://arxiv.org/abs/2211.05102) | Full PDF (18 pp.) |
| Scaling Laws for Neural Language Models | Jared Kaplan et al. | 2020-01-23 (arXiv) | [link](https://arxiv.org/abs/2001.08361) | arXiv metadata/abstract |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Albert Gu; Tri Dao | 2023-12-01 (arXiv) | [link](https://arxiv.org/abs/2312.00752) | Full HTML (main text + references) |


## Systems references

| # | Source | Venue / date | URL | Access |
|---|---|---|---|---|
| <span id="ref-systems-1">1</span> | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al. | arXiv 2205.14135; NeurIPS 2022 | https://arxiv.org/abs/2205.14135 | Full HTML |
| <span id="ref-systems-2">2</span> | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — Dao | arXiv 2307.08691; Jul 2023 | https://arxiv.org/abs/2307.08691 | Full PDF |
| <span id="ref-systems-3">3</span> | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — Shah et al. | arXiv 2407.08608; NeurIPS 2024 | https://arxiv.org/abs/2407.08608 | Full HTML v2 |
| <span id="ref-systems-4">4</span> | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling — Zadouri et al. | arXiv 2603.05451; Mar 2026 | https://arxiv.org/abs/2603.05451 | Full HTML v1 |
| <span id="ref-systems-5">5</span> | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving — Ye et al. | arXiv 2501.01005; MLSys 2025 | https://arxiv.org/abs/2501.01005 | Full PDF |
| <span id="ref-systems-6">6</span> | Orca: A Distributed Serving System for Transformer-Based Generative Models — Yu et al. | OSDI 2022 | https://www.usenix.org/system/files/osdi22-yu.pdf | Full PDF |
| <span id="ref-systems-7">7</span> | Efficient Memory Management for LLM Serving with PagedAttention — Kwon et al. | arXiv 2309.06180; SOSP 2023 | https://arxiv.org/abs/2309.06180 | Full PDF |
| <span id="ref-systems-8">8</span> | SGLang: Efficient Execution of Structured Language Model Programs — Zheng et al. | arXiv 2312.07104; NeurIPS 2024 | https://arxiv.org/abs/2312.07104 | Full PDF v2 |
| <span id="ref-systems-9">9</span> | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve — Agrawal et al. | arXiv 2403.02310; OSDI 2024 | https://arxiv.org/abs/2403.02310 | Full HTML v3 |
| <span id="ref-systems-10">10</span> | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving — Zhong et al. | arXiv 2401.09670; OSDI 2024 | https://arxiv.org/abs/2401.09670 | Full PDF; drafts also link the USENIX version |
| <span id="ref-systems-11">11</span> | Splitwise: Efficient Generative LLM Inference Using Phase Splitting — Patel et al. | arXiv 2311.18677; ISCA 2024 | https://arxiv.org/abs/2311.18677 | Full PDF |
| <span id="ref-systems-12">12</span> | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — Qin et al. | FAST 2025 | https://www.usenix.org/system/files/fast25-qin.pdf | Full PDF |
| <span id="ref-systems-13">13</span> | DeepSeek-V3 Technical Report — DeepSeek-AI | arXiv 2412.19437; Dec 2024 | https://arxiv.org/abs/2412.19437 | Partial HTML, §§3.2–3.4 |
| <span id="ref-systems-14">14</span> | Ring Attention with Blockwise Transformers for Near-Infinite Context — Liu, Zaharia, Abbeel | arXiv 2310.01889; Oct 2023 | https://arxiv.org/abs/2310.01889 | Full HTML; training experiments |
| <span id="ref-systems-15">15</span> | Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD) — He et al. | arXiv 2602.14516; Feb 2026 (v2 Jul 2026) | https://arxiv.org/abs/2602.14516 | HTML |
| <span id="ref-systems-16">16</span> | vLLM V1 alpha release | vLLM blog; Jan 2025 | https://vllm.ai/blog/2025-01-27-v1-alpha-release | Official blog |
| <span id="ref-systems-17">17</span> | vLLM V1 user guide | vLLM docs; Oct 2026 | https://docs.vllm.ai/en/latest/usage/v1_guide/ | Official docs |
| <span id="ref-systems-18">18</span> | Disaggregated prefilling | vLLM docs; Oct 2026 | https://docs.vllm.ai/en/latest/features/disagg_prefill/ | Official docs |
| <span id="ref-systems-19">19</span> | HiCache design | SGLang docs; current | https://docs.sglang.io/docs/advanced_features/hicache_design | Official docs |
| <span id="ref-systems-20">20</span> | Paged attention and in-flight batching scheduler | TensorRT-LLM docs; current | https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html | Official docs |
| <span id="ref-systems-21">21</span> | Disaggregated serving | TensorRT-LLM docs; current | https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html | Official docs |
| <span id="ref-systems-22">22</span> | llama.cpp server README | ggml-org; current | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | Official docs |
| <span id="ref-systems-23">23</span> | Paged attention engine | ONNX Runtime GenAI docs; current | https://github.com/microsoft/onnxruntime-genai/blob/main/docs/paged_attention_engine.md | Official docs |
| <span id="ref-systems-24">24</span> | Introducing NVIDIA Dynamo | NVIDIA developer blog; Mar 2025 | https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/ | Vendor blog |
| <span id="ref-systems-25">25</span> | llm-d announcement | llm-d; May 2025 | https://llm-d.ai/blog/llm-d-announce | Project announcement |


## Algorithmic frontiers references

| # | Source | Date | Link | Access level |
|---:|---|---|---|---|
| 1 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Oct 2022 | https://arxiv.org/abs/2210.17323 | Full text (arXiv HTML) |
| 2 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Jun 2023 | https://arxiv.org/abs/2306.00978 | Full text; MLSys 2024 |
| 3 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | Nov 2022 | https://arxiv.org/abs/2211.10438 | Full text |
| 4 | FP8 Formats for Deep Learning | Sep 2022 | https://arxiv.org/abs/2209.05433 | Full text |
| 5 | DeepSeek-V3 Technical Report | Dec 2024 | https://arxiv.org/abs/2412.19437 | Vendor technical report (full text) |
| 6 | NVFP4 Pretraining | Sep 2025 | https://arxiv.org/abs/2509.25149 | Vendor technical report |
| 7 | Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | 2025-06-24 | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | Vendor blog |
| 8 | NVFP4 Quantization-Aware Distillation (QAD) Report | 2026 | https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf | Vendor technical report |
| 9 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | Feb 2024 | https://arxiv.org/abs/2402.02750 | Full text |
| 10 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | Jan 2024 | https://arxiv.org/abs/2401.18079 | Full text |
| 11 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | May 2024 | https://arxiv.org/abs/2405.04434 | Vendor technical report |
| 12 | Fast Inference from Transformers via Speculative Decoding (Leviathan et al.) | 2023 | https://proceedings.mlr.press/v202/leviathan23a.html | Abstract/bibliographic record only |
| 13 | Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al.) | Feb 2023 | https://arxiv.org/abs/2302.01318 | Full text |
| 14 | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Jan 2024 | https://arxiv.org/abs/2401.10774 | Full text |
| 15 | EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test | Mar 2025 | https://arxiv.org/abs/2503.01840 | Full text |
| 16 | Speculative Speculative Decoding (SSD/Saguaro) | Mar 2026 | https://arxiv.org/abs/2603.03251 | Preprint; OpenReview PDF timed out (504), used arXiv HTML |
| 17 | DeepSeek-V3.2 | Dec 2025 | https://arxiv.org/abs/2512.02556 | Vendor technical report |
| 18 | Mamba-2 / State Space Duality | May 2024 | https://arxiv.org/abs/2405.21060 | Full text |
| 19 | Kimi Linear: An Expressive, Efficient Attention Architecture | Oct 2025 | https://arxiv.org/abs/2510.26692 | Full text; vendor-affiliated |
| 20 | LLaDA: Large Language Diffusion Models | Feb 2025 | https://arxiv.org/abs/2502.09992 | Full text |
| 21 | Mercury Coder technical report; Mercury 2 announcement | Jun 2025; 2026-02-24 | https://arxiv.org/abs/2506.17298; https://www.inceptionlabs.ai/blog/introducing-mercury-2 | Technical report; vendor blog/press. BusinessWire link unavailable; date checked via Yahoo syndication |
| 22 | s1: Simple Test-Time Scaling | Jan 2025 | https://arxiv.org/abs/2501.19393 | Full text |
| 23 | Agentic AI Workload Characteristics | May 2026 | https://arxiv.org/abs/2605.26297 | Preprint |
| 24 | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | May 2024 | https://arxiv.org/abs/2405.16444 | Full text; EuroSys 2025 |
| 25 | NVILA: Efficient Frontier Visual Language Models | Dec 2024 | https://arxiv.org/abs/2412.04468 | Full text; v3 HTML dated 2026-04-25 |


# Verification and Editorial Record {.unnumbered}

Cutoff: 19 October 2026. The user requested delegation through **CommandCode with `deepseek/deepseek-v4.1-flash-fast` and web search**.

## What actually ran

The installed CLI identified itself as CommandCode v1.84.0 and listed the exact requested model. A short probe successfully invoked `web_search` and `web_fetch` for the original FlashAttention paper. Three independent literature sessions were then launched for foundations, systems, and algorithmic frontiers. A separate session audited the synthesis. A systems recovery/editor session and a final synthesis session were delegated to the same model.

All successful invocations used read-only plan permissions, disabled skill discovery and automatic updates, and used `--no-session`. Web search and fetch were available by default; `--tools-enable` is unnecessary for them because that flag only restores tools withheld in headless mode. No alternate model was substituted. Exact per-run model IDs, status, tool counts and provider-reported usage are in [run-manifest.json](../run-manifest.json). Tool requests, responses and errors are in [tool-evidence.jsonl](../tool-evidence.jsonl). Prompts are in [prompts](../prompts/).

The orchestrator initially did independent source checks and drafted a synthesis. After the user clarified that substantive work should stay delegated, further literature synthesis and recovery were assigned to CommandCode. The orchestrator's subsequent role was review, corrections and packaging.

## Evidence levels

- **Primary-source content inspected:** supports the specific method or implementation description read. It does not establish independent replication.
- **Abstract/metadata inspected:** supports title, date, authors and claims actually present in the abstract; it does not imply the full paper was read.
- **Author/project blog or documentation:** supports what its authors report or document, subject to its hardware, workload and version limitations.
- **Derived:** formulas and illustrative arithmetic, with explicit assumptions.
- **Recommended or predicted:** engineering judgments and testable hypotheses, not established experimental results.

The agent bibliographies state their source access. These are agent-reported reading levels, with source-tool evidence retained for inspection. The orchestrator spot-checked central sources; it did not independently reread every cited paper. Neither the agents nor orchestrator reproduced engine benchmarks or trained/evaluated a model in this task. This is a selective review rather than a systematic survey with exhaustive inclusion criteria.

## Audit disposition

The separate [audit memo](../audit.md) found no arithmetic errors in the illustrative KV/weight-memory calculations and no fabricated citations among its checked sources. That statement applies only to the audited subset.

Corrections accepted in the reviewed report and supplied to the synthesis agent:

1. Approximate Speculative Decoding is an approximate **greedy-verification** proposal; it must not be represented as a drop-in stochastic rejection sampler.
2. Exact speculation preserves a target distribution under assumptions and numerical limitations, not necessarily an identical random sample for the same seed.
3. vLLM's adaptive-verification article rejects sampled-token-dependent confidence features and uses a distribution-level feature; the source does not claim to ship the biased alternative.
4. Adaptive verification changes the number of proposed tokens verified. It need not reduce already-paid drafting work and requires compatible variable-length verification.
5. Push-based KV transfer requires earlier decode placement; the source describes reasons to retain pull mode, including routing flexibility and bidirectional transfer.
6. Decode-retained reusable state includes prompt KV as well as generated-token KV.
7. Publication and source-content checks are distinct from validation of benchmark results.

One auditor suggestion was **not adopted as a general claim**: adding an upper bound of compute time plus communication time for real execution. That sum can describe an idealized two-resource model; it does not bound arbitrary runtime overhead, additional resources, dependencies or synchronization. The report retains the explicitly optimistic lower bound and its limitations.

The auditor's criticism of “abstract-only” wording was interpreted as a need to clarify **our access level**, not as evidence that the paper itself lacks full experimental sections. The agent subsequently inspected additional experimental material. Newer source additions made after the initial audit are not implicitly covered by that audit.

## Runtime limitations

The systems and frontier research sessions collected their sources but ended with API-server errors while producing final answers. Their preserved streams contained overlapping drafts; fresh same-model editors consolidated them. Published memos use the completed final response from the successful original or recovery run, with review corrections. Raw outputs are retained separately from edited deliverables; verbose NDJSON logs are losslessly compressed and their round-trip SHA-256 hashes checked before removing the uncompressed copies.

The [second review](../review_memos.md) checked overstated runtime comparisons and formula assumptions. Editorial corrections include restricting the 12Ld² parameter estimate to its architecture assumptions; identifying batch weight-traffic amortization correctly; qualifying incomplete prefill traffic estimates and recurrent-state accounting; acknowledging llama.cpp RPC, TensorRT-LLM's PyTorch backend and decode context parallelism; and distinguishing default preemption from explicit offload. The synthesis's missing `d` in `4LBnd` and its `log(q_max)`/`logit(q_max)` transcription error were also corrected. The original returned memos remain in raw evidence.

An attempted recovery launch with `--effort medium` was rejected because this CLI supports `off`, `low`, `high`, and `max`. A subsequent `--effort low` attempt tried to persist global configuration and failed under filesystem permissions. Recovery proceeded with the original working command and no effort override; no permission bypass or global configuration change was made.

Future hypotheses are proposed experiments. Hardware selection, exact checkpoint support, quality targets and latency requirements remain design inputs to choose before implementing the engine.


## Foundation source-access limits

This is a selective review, not exhaustive. Twelve sources were read only as abstracts or metadata and are marked in the table; claims from those are limited to what the abstract states. I did not benchmark anything, and I deliberately omit speedup figures. Rosenblatt’s full text could not be fetched: the UIC PDF, the Henriques mirror, and the S3 mirror all timed out, and the APA record rendered only a loading shell. LSTM’s MIT Press page timed out; I used the CMU PDF. Nature’s full text is paywalled; I read the abstract. XLA and ONNX Runtime are living documentation, not peer-reviewed papers. The historical narrative is my synthesis: individual claims are sourced, but the causal story is not a claim of inevitability. I attribute the architecture to Vaswani et al., and the cache variants to Shazeer, Ainslie et al., and DeepSeek-AI, rather than to any serving library.


## Audit documents

The original [audit](../audit.md) and [second review](../review_memos.md) are available alongside the [research manifest](../run-manifest.json) and [source-tool evidence](../tool-evidence.jsonl). Audit suggestions rejected or qualified by the verification record are not treated as accepted facts. Raw traces and duplicate drafts are cataloged in the complete source inventory below rather than repeated as textbook chapters.
