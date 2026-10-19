# Neural Inference Engines: Foundations, Systems, and a Build Plan

Research synthesis, cutoff **19 October 2026**. Literature research, source audits, recovery editing and synthesis were delegated to CommandCode agents using **deepseek/deepseek-v4.1-flash-fast**. The orchestrator reviewed and corrected the results. The systems and frontier sessions collected evidence but hit API errors during final output; same-model recovery agents consolidated their preserved drafts. Companion memos: [foundations](foundations.md), [systems](systems.md), [frontiers](frontiers.md). See [verification and limitations](VERIFICATION.md). This is a selective review, not an exhaustive survey or benchmark reproduction. Publication/content checks do not imply independent replication.

## 1. Origins and first principles

Neural inference here means forward evaluation of a trained parameterized function, not Bayesian or statistical inference. The historical chain explains the runtime’s shape. McCulloch and Pitts treated the neuron as a threshold logic unit, so early “inference” was logical evaluation of a fixed circuit ([1943](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)). Rosenblatt’s perceptron moved knowledge into learned weights; the full text was not available here, so this is an abstract-level claim ([1958](https://psycnet.apa.org/record/1959-09865-001)). Rumelhart, Hinton, and Williams established backpropagation as the modern training recipe, making the forward graph a parameterized numeric function ([1986](https://www.nature.com/articles/323533a0)). Shannon framed language as next-symbol prediction ([1951](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)); Bengio et al. made that predictor a neural network with distributed representations ([2003](https://jmlr.org/papers/v3/bengio03a.html)).

Recurrence introduced persistent state: LSTM carried a fixed-size hidden state ([1997](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf)), seq2seq encoded once and decoded token by token ([2014](https://arxiv.org/abs/1409.3215)), and Bahdanau attention let each decode step inspect the source ([2015](https://arxiv.org/abs/1409.0473)). The Transformer replaced recurrence with attention ([2017](https://arxiv.org/abs/1706.03762)); GPT-1 was an early influential decoder-only causal language model ([2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)), and GPT-3 made inference a first-order systems cost ([2020](https://arxiv.org/abs/2005.14165)). MQA, GQA, and MLA then turned the KV cache into an explicit architectural choice ([MQA](https://arxiv.org/abs/1911.02150), [GQA](https://arxiv.org/abs/2305.13245), [MLA](https://arxiv.org/abs/2405.04434)).

Two first principles follow. First, a neural engine executes a fixed numeric graph whose topology is largely query-independent; classical symbolic inference searches a query-dependent proof or rule graph, so its cost is measured in rule applications rather than FLOPs. Second, decoder inference is stateful and sequential: **prefill** processes prompt tokens in parallel and writes reusable state; **decode** extends each sequence, usually one token per iteration, and reads that state. Training is parallel over positions and computes gradients; inference is forward-only and carries persistent cache state. That is why the same model can be compute-bound in prefill and bandwidth-bound in decode.

## 2. The cost model, with units

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

## 3. Literature map and the distinctions that matter

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

## 4. Verified 2025–October 2026 snapshot

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

## 5. Recommended architecture and build order

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

## 6. Realistic benchmark methodology

**Recommendation.** Define metrics before measuring. TTFT is request arrival to first streamed token; state whether tokenization, preprocessing, queueing, and network are included. TPOT is \((t_{\text{last}}-t_{\text{first}})/(N_{\text{out}}-1)\), but also report the full inter-token-latency distribution. SLO-goodput is the maximum request rate meeting both TTFT and TPOT/ITL targets at a stated attainment level, per GPU. Report p50/p95/p99, errors, rejected requests, offered versus achieved load, output tokens/s, requests/s, cost per million tokens, and energy per token where relevant.

Use open-loop arrivals—a fixed timestamp trace or bursty stochastic process—not a closed-loop client that throttles on latency. Poisson is a weak default for production; include burstiness, multi-turn/agentic interleaving, and prefix-reuse skew. Warm up, measure steady state, and separate queueing from execution. Cross input/output lengths, shared and independent prefixes, cold/warm caches, low-to-saturating concurrency, several context lengths, and cancellation. Compare disaggregation at the **same GPU count and SLO attainment**, not against a colocated system given more resources.

Pin engine commit, CUDA/driver, GPU SKU, clocks and power limits, model/tokenizer revision, quantization, parallelism, block size, max batch tokens, chunk size, scheduling policy, cache state, seed, trace hash, warmup, and run count. Publish raw latency samples and keep kernel microbenchmarks separate from end-to-end serving. Include ablations and failure injection: transfer timeout, prefill/decode crash, cache-backend outage, overload/429, and cancellation storms. Correctness tests should cover causal masks, padding/ragged batches, GQA head mapping, positional offsets, cache append/fork/truncate, block boundaries, EOS/stop behavior, all-accepted and all-rejected speculation, numerical stability, and quantization quality. Stochastic sampling needs distribution-oriented tests, not only fixed-seed string equality.

## 7. Five falsifiable future hypotheses

These are **predictions/proposals**, not established findings.

| Hypothesis | Controlled experiment | Success metric | What would falsify it |
|---|---|---|---|
| Joint scheduling of prefill chunks and speculative verification beats fixed independent settings | Fixed vs adaptive policies on the same mixed trace | Higher SLO-goodput at equal quality and hardware | Controller overhead or instability outweighs savings |
| A typed per-layer state interface supports attention, compressed, and recurrent layers without material abstraction cost | Same model/kernels behind specialized vs common state APIs | Equivalent numerics and small measured overhead; reuse across architectures | The abstraction prevents critical fusion/layout choices |
| Reuse-and-transfer-cost-aware cache placement beats recency/LRU | Replay multi-turn agent traces at fixed capacity and bandwidth | Lower p95 TTFT and recompute FLOPs at stable hit rate | Prediction errors cause churn and evict useful prefixes |
| Quality-budgeted per-component precision improves task-success per joule | Sweep weight/activation/KV precision at fixed task-quality threshold | Lower energy and latency at matched task success | Error accumulates on long context or reasoning, or QAD erases the gain |
| Session-aware routing reduces agent completion cost versus load-only routing | Same multi-step tasks under locality-aware vs load-only routing | Lower cost and completion latency at equal success | Locality concentrates load and increases queueing |

## 8. Ordered reading path

1. **Foundations:** McCulloch–Pitts (1943), Rosenblatt (1958), Rumelhart et al. (1986), Shannon (1951), Bengio et al. (2003), LSTM (1997), seq2seq (2014), Bahdanau et al. (2015), Transformer (2017), GPT-1 and GPT-3.
2. **Mechanics:** Pope et al. on prefill/decode ([2022](https://arxiv.org/abs/2211.05102)), the roofline paper, MQA → GQA → MLA, FlashAttention, then Mamba.
3. **Serving systems:** Orca, PagedAttention/vLLM, SGLang, Sarathi-Serve, DistServe/Splitwise/Mooncake, FlashInfer, and the vLLM V1 and HiCache documentation.
4. **Algorithms:** GPTQ/AWQ/SmoothQuant, FP8/NVFP4, KIVI/KVQuant, then exact speculative decoding (Leviathan/Chen), Medusa, EAGLE-3, SSD, and ASD.
5. **Workloads:** the agent-workload characterization, CacheBlend, Kimi Linear, LLaDA/Mercury, and s1.
6. **2026 sources:** FA4, SGLang DeepSeek-V4, the vLLM disaggregation/bidirectional/push/adaptive-verification posts, ReplaySSM, SuffixReplay, and AMPD—then re-derive the cost model against your own hardware.
