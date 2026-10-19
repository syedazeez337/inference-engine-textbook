# Foundations of Neural Inference: From Threshold Logic to Autoregressive Decoders

*Research memo — 19 October 2026. Scope: forward execution of trained neural models, especially decoder-only Transformers. Serving scheduling and speculative decoding are excluded as separate tracks. “Inference” here means model execution, not Bayesian posterior inference.*

## 1. Evidence status

I used 22 sources. The bibliography marks what I actually read. Twelve are abstract/metadata-only — Rosenblatt, Rumelhart, Bengio, Sutskever, Bahdanau, Vaswani, GPT-3, MQA, GQA, FlashAttention, TVM, and Kaplan — and I limit claims from them accordingly. Rosenblatt is metadata-only because three PDF endpoints timed out. The MIT Press LSTM page also timed out, so I read the CMU PDF. arXiv abstract pages returned only titles, so I used the arXiv API for dates and abstracts. I report no speedup figures: none of the benchmarks I read could be compared on identical hardware, workload, comparator, and metric. Sections 2 is source findings; Sections 3–7 are my derivations and synthesis; Sections 8–9 are recommendations.

## 2. Source findings: the chain

### 2.1 Threshold logic and learning

McCulloch and Pitts (1943) modeled the neuron as an all-or-none threshold unit and showed that networks of such units can be described by propositional logic. The core calculus assumes a fixed net; learning appears only as an extension. In that framing, inference is evaluation of a logical circuit, not statistical prediction. ([paper](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf))

Rosenblatt (1958) introduced the perceptron as a probabilistic model of information storage and organization, with a learning procedure and predicted learning curves. This is the turn from hand-specified logic to parameters fitted from data. I did not verify the convergence theorem or full model. ([record](https://psycnet.apa.org/record/1959-09865-001))

Rumelhart, Hinton, and Williams (1986) is the widely cited modern statement of back-propagation: repeated weight adjustment to minimize output error, with hidden units learning task features. Caveat: the 1986 letter popularized the recipe; it was not the first derivation of reverse-mode differentiation. I did not verify the earlier lineage here. ([paper](https://www.nature.com/articles/323533a0))

### 2.2 Probabilistic language models

Shannon (1951) framed next-symbol prediction as a way to estimate the entropy and redundancy of English, using n-gram approximations and human prediction experiments. This is a foundational milestone, not the sole origin of sequence modeling; Shannon's 1948 work already discusses language approximations. ([1951 paper](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf), [1948 paper](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf))

Bengio et al. (2003) replaced sparse n-gram counts with a neural network over distributed word representations, explicitly to fight the curse of dimensionality. The model learns the joint probability of a sequence and generalizes to unseen sequences through similarity in representation space. Inference is a forward pass plus a normalization over the vocabulary. ([abstract](https://jmlr.org/papers/v3/bengio03a.html))

### 2.3 Recurrence, encoder-decoder, attention

Hochreiter and Schmidhuber (1997) introduced LSTM to preserve error flow over long lags through a constant error carousel and multiplicative input/output gates. Update complexity is O(1) per time step and weight, and the recurrent state is fixed-size. That is a key inference property: memory does not grow with sequence length, but the state must compress the past. ([paper](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf))

Sutskever, Vinyals, and Le (2014) used a multilayered LSTM to map an input sequence to a fixed-dimensional vector and another LSTM to decode the target one token at a time. The fixed vector is an explicit bottleneck; decoding is inherently sequential. ([abstract](https://arxiv.org/abs/1409.3215))

Bahdanau, Cho, and Bengio (2015) removed that bottleneck by letting the decoder soft-search over encoder states. Attention scores are computed against all source positions, so inference cost grows with source length, but the decoder is no longer forced through one vector. ([abstract](https://arxiv.org/abs/1409.0473))

### 2.4 Transformer and decoder-only inference

Vaswani et al. (2017) replaced recurrence with self-attention, enabling parallel training over sequence positions while keeping an encoder-decoder structure and a causal mask in the decoder. The architectural change is decisive for inference: training can process all tokens at once, but decoder inference still generates one token at a time. ([abstract](https://arxiv.org/abs/1706.03762))

GPT-1 (Radford et al., 2018) adapted the Transformer decoder into a 12-layer, 768-dimensional, 12-head decoder-only model trained with a next-token objective and then fine-tuned. GPT-3 (Brown et al., 2020) scaled the same autoregressive recipe to 175B parameters and demonstrated few-shot behavior without gradient updates. Once models are this large, inference cost and latency become first-order concerns. ([GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-3](https://arxiv.org/abs/2005.14165))

### 2.5 Compressing the inference state

Shazeer (2019) observed that incremental decoding is limited by repeatedly loading key and value tensors, and proposed multi-query attention: all query heads share one key/value head. The KV cache shrinks by roughly the number of query heads, at some quality cost. ([paper](https://arxiv.org/abs/1911.02150))

Ainslie et al. (2023) generalized this to grouped-query attention, with an intermediate number of KV heads, and showed existing multi-head checkpoints can be uptrained into GQA. GQA is the common compromise: fewer KV heads than query heads, but more than one. ([paper](https://arxiv.org/abs/2305.13245))

DeepSeek-V2 (DeepSeek-AI, 2024) introduced multi-head latent attention: keys and values are jointly compressed into a latent vector, and a decoupled rotary key is cached separately. The paper gives the cache as (d_c + d_h^R) per token per layer, not 2× the latent. For DeepSeek-V2, d_c=512, d_h^R=64, d_h=128, n_h=128, L=60, giving 576 elements per token per layer and 34,560 elements per token — equivalent to GQA with 2.25 groups, as the paper states. ([paper](https://arxiv.org/abs/2405.04434))

Mamba (Gu and Dao, 2023) takes the other route: a selective state-space model with a fixed-size recurrent state, constant-time per-step inference, and no KV cache. The paper explicitly frames the tradeoff as state compression: attention stores the entire context; recurrent models compress it. ([paper](https://arxiv.org/abs/2312.00752))

## 3. Why inference differs from training

Training and inference share the forward graph, but the surrounding computation differs.

- Training needs gradients. Backward roughly doubles matmul cost per token; a common estimate is 6PD FLOPs for P parameters and D tokens (2 forward, 4 backward). Inference is forward-only, about 2P FLOPs per token plus attention.
- Training parallelizes over tokens via teacher forcing; decoder inference cannot, because token t+1 depends on token t.
- Training stores activations for backward; inference frees them after each layer and keeps only the KV cache or recurrent state across steps.
- The bottleneck moves. Large-batch training is compute-bound; single-sequence decode is memory-bandwidth-bound because each token reads all weights and the whole KV cache.
- The objective changes: training minimizes loss; inference optimizes latency, throughput, memory capacity, and tail behavior.
- Precision changes: inference can quantize weights, activations, and KV cache; training is more sensitive to gradient precision and stability.

An inference engine is therefore not “training minus backward.” It is a stateful, latency-sensitive runtime whose cost is dominated by memory traffic and cache layout.

## 4. Why inference differs from symbolic inference

Classical symbolic inference operates on discrete symbols and rules: unification, resolution, forward/backward chaining, proof search. The computation is query-dependent; the graph of rule applications changes with the goal, and cost is governed by branching factor and search depth. Correctness is exact relative to the rule system.

Neural inference operates on dense tensors and learned weights. The graph is fixed, modulo dynamic shapes and control flow, and knowledge is distributed across parameters. Correctness is statistical and approximate. Cost is governed by FLOPs, bytes moved, and parallelism, not search depth. McCulloch and Pitts are the hinge: their threshold nets can realize propositional logic, so early neural computation looked like logic; learning and continuous activations turned the program from symbol manipulation into parameterized numeric evaluation. Modern LLM inference is still a computation graph, but a compiled and reused one — which is why compilers and kernel libraries matter.

## 5. Inference as a computation graph

A trained model is a DAG of tensor operations, with weights as constants and activations as edges. Engines specialize it: fold constants, fuse elementwise ops, choose layouts, plan memory buffers, and dispatch kernels. TVM exposes graph-level and operator-level optimization, including fusion and auto-tuned schedules across backends. XLA compiles StableHLO/HLO with target-independent and target-specific fusion, shape specialization, buffer analysis, and partitioning. ONNX Runtime applies graph optimizations, partitions subgraphs across execution providers, and wraps each partition as a fused operator with a provider-specific allocator. ([TVM](https://arxiv.org/abs/1802.04799), [XLA](https://openxla.org/xla/architecture), [ONNX Runtime](https://onnxruntime.ai/docs/reference/high-level-design.html))

Two details matter for Transformers:

- **Tensor layout.** Activations are commonly [B, T, d]; weights [out, in]; KV cache [B, H, T, d_h] or [B, T, H, d_h]. The choice determines coalescing, transpose cost, and whether attention can read K/V contiguously. Layout assignment is a compiler decision, not cosmetic.
- **State.** Autoregressive decoding unrolls the same graph per token, with a growing KV cache as mutable state. The graph is static; the cache length is dynamic. Engines specialize on shape buckets or use dynamic-shape kernels, and must keep cache layout consistent across prefill and decode.

CPU and GPU execution differ mainly in the memory hierarchy. CPUs rely on caches, SIMD, and multithreading; GEMV is often bandwidth-bound, and weight quantization helps. GPUs have HBM plus on-chip SRAM and tensor cores; GEMM tiles well, but decode underutilizes compute and is sensitive to kernel-launch overhead and KV-cache traffic. The same roofline applies with different ridge points.

## 6. Roofline, GEMM/GEMV, prefill/decode

The roofline model bounds attainable performance by min(peak FLOP/s, bandwidth × operational intensity), where operational intensity is FLOPs per byte of DRAM traffic and the ridge point is peak FLOP/s divided by bandwidth. Williams, Waterman, and Patterson define intensity between caches and memory, not between processor and cache. ([paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf))

This gives the central split:

- **GEMV** (BLAS Level 2) computes y = Ax. For large A, each weight is used once per output element, so intensity is about 2/b FLOP/byte for b bytes per element. It is memory-bound.
- **GEMM** (BLAS Level 3) computes C = AB. Tiling reuses both operands, so intensity grows with tile size. Decode with batch B is a skinny GEMM; increasing B raises intensity until compute-bound.

Pope et al. (2023) name the phases: **prefill** processes all prompt tokens in parallel; **decode** generates output tokens autoregressively. They also state the baseline that an N-parameter decoder-only model needs 2N matmul FLOPs per token in the forward pass. Prefill is a large GEMM and can reach high model-FLOPs utilization; decode is a sequence of GEMV/skinny GEMMs dominated by loading weights and KV cache. ([paper](https://arxiv.org/abs/2211.05102))

FlashAttention is the clearest inference-relevant optimization that changes memory traffic rather than asymptotic FLOPs: it tiles attention to keep intermediates in on-chip SRAM and reduces HBM reads/writes, while remaining exact. It helps prefill (long attention matrices) and decode (KV-cache reads), but does not make attention free. ([paper](https://arxiv.org/abs/2205.14135))

Caveats: roofline is an upper bound, not a prediction; it ignores latency, occupancy, tail effects, and multi-chip communication. Attention FLOPs are often small relative to the MLP for large models, but the KV cache can dominate memory capacity and bandwidth.

## 7. Deriving the cost model

### 7.1 Assumptions

L layers; model width d; h query heads; d_h = d/h; g KV heads; vocabulary V; d_ff = 4d; b bytes per element (2 for FP16/BF16); batch B; context n; prompt length T. Ignore biases, layer norms, RoPE, softmax, sampling, padding, and allocator overhead. Count one multiply-add as 2 FLOPs.

### 7.2 Parameters and FLOPs

For the MHA special case with a non-gated 4d-width MLP, per-layer attention projections Q, K, V, O = 4d²; MLP = 2·d·4d = 8d²; total = 12d². So:

**P ≈ 12Ld² + Vd** (tied embeddings; untied adds another Vd).

More generally for ordinary GQA/MQA projections and an m-matrix MLP, **P ≈ L(2d² + 2g d²/h + m·d·d_ff) + Vd**, where m=2 for a non-gated MLP and m=3 for a gated MLP. MLA and MoE need their own parameter accounting. ([GLU variants](https://arxiv.org/abs/2002.05202))

For GPT-3, L=96, d=12288: 12·96·12288² ≈ 173.9B, plus embeddings ≈ 174.6B, consistent with the reported 175B. ([paper](https://arxiv.org/abs/2001.08361))

Forward FLOPs per token: **2P** for matmuls, plus attention:
- Decode at context n: **4Lnd** (QKᵀ and AV, each 2nd per layer).
- Prefill T tokens: **4LT²d** dense; causal kernels can roughly halve the score/AV term.

Training, for D tokens: **C_train ≈ 6PD** (forward 2P, backward 4P per token). This is a derivation consistent with Kaplan et al.’s compute-scaling framing, not a measured constant.

### 7.3 KV cache

For MHA, each token per layer stores K and V, each of width h·d_h = d:

**M_MHA = 2Lnd elements; bytes = 2Lndb.**

For GQA with g KV heads:

**M_GQA = 2Lngd_h elements; bytes = 2Lngd_hb.** Ratio to MHA = g/h.

For MQA, g=1: **M_MQA = 2Lnd_h elements**; ratio = 1/h.

For MLA, cache the compressed latent and the decoupled rotary key:

**M_MLA = (d_c + d_R)Ln elements.**

No factor of 2: the latent jointly represents K and V; the rotary key is shared across heads. DeepSeek-V2: d_c=512, d_R=64, L=60, so 576·60 = **34,560 elements per token**; at 2 bytes, **69,120 bytes per token**. This is linear in n, not constant.

For a state-space model, cache the recurrent state, not K/V:

**M_state = L · d_inner · N elements** describes the core recurrent state in this simplified model, independent of n. Actual implementations also retain convolution buffers and possibly checkpoint/replay state; obtain their shapes and dtypes from the checkpoint and runtime rather than treating this expression as total runtime memory.

Batch multiplies all of these; precision scales bytes by b; padding and allocator fragmentation are extra.

### 7.4 Hand-checkable example

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

## 8. Ranked reading path

**Tier 1 — mechanics before writing a kernel:** Vaswani et al. 2017; Pope et al. 2023; Williams et al. 2009; Shazeer 2019; Ainslie et al. 2023; DeepSeek-AI 2024; Dao et al. 2022.

**Tier 2 — runtime and compilers:** Chen et al. 2018 (TVM); OpenXLA architecture docs; ONNX Runtime architecture docs; Kaplan et al. 2020.

**Tier 3 — historical grounding:** McCulloch and Pitts 1943; Rosenblatt 1958; Rumelhart et al. 1986; Shannon 1951; Bengio et al. 2003; Hochreiter and Schmidhuber 1997; Sutskever et al. 2014; Bahdanau et al. 2015; Radford et al. 2018 (GPT-1); Brown et al. 2020 (GPT-3).

**Tier 4 — the alternative state model:** Gu and Dao 2023 (Mamba).

## 9. What to implement first

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

## 10. Limits and access failures

This is a selective review, not exhaustive. Twelve sources were read only as abstracts or metadata and are marked in the table; claims from those are limited to what the abstract states. I did not benchmark anything, and I deliberately omit speedup figures. Rosenblatt’s full text could not be fetched: the UIC PDF, the Henriques mirror, and the S3 mirror all timed out, and the APA record rendered only a loading shell. LSTM’s MIT Press page timed out; I used the CMU PDF. Nature’s full text is paywalled; I read the abstract. XLA and ONNX Runtime are living documentation, not peer-reviewed papers. The historical narrative is my synthesis: individual claims are sourced, but the causal story is not a claim of inevitability. I attribute the architecture to Vaswani et al., and the cache variants to Shazeer, Ainslie et al., and DeepSeek-AI, rather than to any serving library.

## Bibliography

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
