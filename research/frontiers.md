# Algorithmic Frontiers in Inference Engines: Consolidated Research Memo

**Date:** 2026-10-19  
**Status:** Editorial consolidation of the source agent’s partial drafts. I did not independently read these sources. Access levels below describe what the source agent reported, including abstract-only items and access failures. I have removed repeated material and all numerical benchmark speedups, and corrected five recurring errors: exact stochastic speculation preserves the target distribution rather than producing identical random samples; quantization changes the target; hardware- and workload-specific gains are not universal; 2022–2024 work was not confined to weight precision; and vendor research is not independent evidence.

## 1. Three budgets and two correctness axes

The useful framing is not a chronology of “weights, then everything else,” but three interacting budgets: **weight/activation bytes**, **KV/state residency**, and **step/scheduling cost**. A low-bit weight model can remain KV-bound at long context; a speculative decoder can hide draft latency but cannot create KV capacity; and an agent workload is decode-dominated only while its cached state survives.

The drafts’ claim that 2022–2024 was “mostly about weights” is too narrow. KV-cache quantization, latent attention, tree verification, and serving scheduling were already first-class concerns in the reviewed sources. Weight quantization was the most standardized and easiest to benchmark, not the only active problem. Attention layout, state growth, and request scheduling already constrained real deployments.

The second axis is correctness. Correct rejection-sampled speculation and compatible exact prefix reuse preserve target semantics subject to numerical limits. Weight/KV quantization and approximate acceptance change the deployed target or sampling rule. Native sparse, linear-attention and diffusion checkpoints are alternative architectures, not automatically approximate executions of a dense autoregressive checkpoint. Their runtime can faithfully execute their own semantics; comparisons require separate task-quality evaluation.

## 2. Quantization: mechanisms and limits

**GPTQ** treats layer-wise weight quantization as a second-order reconstruction problem. It quantizes weights in blocks, uses an approximate Hessian inverse, and updates remaining weights to compensate for rounding error. Its limitation is structural: it is weight-only, reduces memory traffic rather than arithmetic, does not quantize activations, and does not improve compute-bound batched execution. Its quality claims are calibration- and model-specific; lower bit-widths require smaller groups and still degrade.

**AWQ** identifies salient channels by activation magnitude rather than weight magnitude and protects them with per-channel scaling. It avoids backpropagation and reconstruction, which the authors argue reduces calibration overfitting. It remains W4A16 weight-only: speed depends on packing and fused dequantization kernels, and it does not make arithmetic cheaper. Its robustness advantage is relative, not universal.

**SmoothQuant** addresses the activation-outlier problem that blocks W8A8. Outliers persist in fixed channels, so per-tensor activation quantization wastes effective bits. SmoothQuant applies a mathematically equivalent per-channel scaling that migrates quantization difficulty from activations to weights. It preserves accuracy on several large models under INT8 GEMM, but it is an 8-bit solution tied to per-tensor static quantization; it is not a path to 4-bit activations.

**FP8** defines E4M3 and E5M2 with different range/precision tradeoffs and recommends E4M3 for weights and activations, E5M2 for gradients. The original work was largely simulated, not a hardware speed demonstration. DeepSeek-V3 is stronger production evidence: a large MoE trained in FP8 with fine-grained scaling and frequent FP32 promotion because H800 accumulation retained limited precision. That is a co-designed hardware/software result, not proof that naive FP8 works everywhere. FP8 inference still depends on tensor-core generation, scaling granularity, and accumulation behavior.

**FP4** is more aggressively hardware-specific. NVFP4 uses E2M1 elements, small micro-blocks, per-block E4M3 scales, and a per-tensor FP32 scale. NVIDIA’s pretraining result required random Hadamard transforms, two-dimensional weight scaling, stochastic rounding on gradients, and keeping a minority of linear layers in BF16; late BF16 switching closed most of the remaining loss gap. PTQ works better on very large models than on small or sensitive post-trained models, where quantization-aware distillation is often needed. QAD is a training procedure, not a free conversion. All of these are vendor technical reports, not independent reproductions.

**KV quantization** has a different failure surface. KIVI observes that key caches have persistent per-channel outliers while value caches do not, so it quantizes keys per-channel and values per-token and keeps a recent FP16 residual window. It degrades on multi-query-attention models, which may need higher precision. KVQuant adds pre-RoPE key quantization, non-uniform datatypes, sparse outlier isolation, and attention-sink handling; it requires calibration, custom kernels, and fused RoPE. MLA changes the outlier structure entirely, so per-channel assumptions do not automatically transfer. DeepSeek-V2 already deploys MLA with roughly 6-bit average KV quantization.

The central correction is that **quantization changes the target distribution**. Exact speculative decoding does not undo quantization error: it verifies the quantized model’s distribution. Quality deltas are small in many reported settings but nonzero, and they are largest on reasoning, long context, small models, and architectures whose outlier structure differs from the calibration assumptions.

## 3. Speculative decoding: exact versus approximate

Leviathan et al. and Chen et al. independently introduced draft-and-verify decoding. A draft proposes tokens; the target scores them in parallel; a modified rejection rule accepts a prefix and resamples from the residual distribution. The guarantee is **distribution preservation**, not sample-for-sample identity. Under stochastic sampling, exact speculative decoding produces different random samples that follow the same target distribution; “identical outputs” language applies only to greedy decoding or within hardware numerics. The drafts’ phrasing on this point was imprecise.

Exact speculation often targets latency, but can also improve throughput in suitable regimes. Its benefit depends on acceptance, target/draft cost, verification work, batch size and scheduling. At high load, additional verification can consume scarce compute and reduce goodput; this must be measured rather than assumed for every large batch.

**Medusa** replaces the separate draft model with extra decoding heads and tree attention. Its standard rejection mode can be exact, but its default **typical acceptance** selects “reasonable” candidates using an entropy threshold and is explicitly not distribution-preserving. That is an approximate quality/speed trade, not lossless acceleration. Tree size has diminishing returns, and at sufficiently large batch the linear layers become compute-bound.

**EAGLE-3** predicts tokens directly, fuses low-, mid-, and high-level target features, and simulates multi-step drafting during training. It uses strict speculative acceptance, so it is lossless with respect to its target, but it requires training a draft per target and its gains are task- and benchmark-specific.

The 2026 development is **Speculative Speculative Decoding (SSD/Saguaro)**. It places the draft on a separate device, predicts likely verification outcomes while verification runs, and pre-speculates for them in a cache; a hit removes drafting latency. It is lossless in principle, but it consumes an extra device, cache-hit rate falls at larger batch or higher temperature, and it remains a preprint rather than a verified deployment. The paper itself notes that speculative decoding is generally ineffective for throughput-bound workloads such as large-scale RL rollouts. The 2026 claim should therefore be treated as an interesting prototype, not independent confirmation.

## 4. MoE, MLA, hybrid/SSM, and diffusion

**MoE** reduces FLOPs per token, not the memory required to hold all experts. At batch-one decode, active-expert weights and KV cache dominate bandwidth; at large expert parallelism, all-to-all communication becomes the bottleneck. DeepSeek-V2 and V3 make this concrete: fine-grained routed experts plus shared experts, device- or node-limited routing, auxiliary or bias-based balancing, and prefill/decode disaggregation. Their throughput figures are vendor-reported and tied to specific clusters.

**MLA** compresses KV by caching a low-rank latent plus a decoupled RoPE key. It changes the KV layout and therefore which quantization schemes transfer. **DeepSeek Sparse Attention** reduces main-attention compute by selecting a fixed top-k of KV entries, but the indexer itself remains quadratic and the full KV cache is still stored. Sparse attention is not a KV-memory solution.

**Mamba-2/SSD** shows the duality between structured state-space models and masked attention. Its state is constant-size, so decode cost does not grow with context, but that state is lossy for exact retrieval and copying. **Kimi Linear** is the strongest reviewed evidence that a hybrid can match full attention under a matched recipe: it interleaves a gated delta-rule linear layer with MLA at a 3:1 ratio and keeps a minority of full-attention layers. It still relies on those full-attention layers for retrieval, and fixed-state linear layers remain weaker at exact recall.

**Diffusion language models** trade sequential decode for parallel refinement. LLaDA is bidirectional and therefore incompatible with causal KV caching; it is competitive on some benchmarks but weaker on others, sensitive to sampling hyperparameters, and needs semi-autoregressive remasking for instruct behavior. Mercury commercializes diffusion for code and reports strong single-shot throughput, but Mercury 2 is a vendor blog/press claim with no technical paper or open weights. Diffusion is a better fit for single-shot, latency-sensitive generation than for multi-turn agents unless block-causal or hybrid caching is added.

## 5. Test-time compute, agents, and multimodal serving

**Test-time compute.** s1 shows that a small curated reasoning set plus “budget forcing”—suppressing the end-of-thinking token and appending “Wait,” or forcing termination—can induce test-time scaling cheaply. The effect flattens with extended budgets and is bounded by context length; rejection sampling can even scale inversely because longer samples are often the ones that went wrong. Reasoning workloads are therefore decode-heavy and long-output, which favors KV efficiency and low-batch exact speculation but works against speculation in throughput-bound RL rollouts.

**Agents.** The 2026 agent characterization is the most directly relevant systems evidence, but it is a preprint using two open models and one scaffold. Its structural findings are stronger than its exact percentages: turn counts are long-tailed; SWE-style tasks accumulate large contexts; with an ideal cache, most input tokens are prefix hits; decode dominates LLM time; and failed runs carry larger contexts. The authors stress that this is a no-thrashing regime. Once aggregate agent context exceeds GPU memory, eviction turns a decode-dominated workload back into expensive recomputation. Capacity planning should target resident KV bytes and eviction rate, not average prompt length.

**CacheBlend** extends reuse beyond the exact prefix by reusing precomputed chunk KV caches and selectively recomputing a small fraction of tokens to restore cross-chunk attention. It is transformer-specific and evaluated on RAG/QA chunks rather than agent traces, but the mechanism is directly relevant to persistent agent state.

**Multimodal.** NVILA’s “scale-then-compress” recipe increases spatial/temporal resolution and then compresses visual tokens through spatial-to-channel reshaping and temporal averaging. It reduces visual-token load and training cost, but accuracy is competitive rather than uniformly better, and visual-token KV behavior differs from text-only prefix reuse. Its efficiency claims are hardware- and configuration-specific.

## 6. Five falsifiable hypotheses

**H1 (state over FLOPs).** For long-horizon agent traces with high prefix-cache hit rates and no KV thrashing, an intervention that reduces attention FLOPs but leaves resident KV bytes unchanged will produce a smaller end-to-end latency improvement than one that reduces resident KV bytes by a comparable factor. Falsified if compute-only attention sparsification consistently wins across models and traces.

**H2 (speculation ceiling).** Exact speculative decoding is a latency optimization whose benefit shrinks as batch size and target compute intensity rise; increasing draft acceptance cannot compensate once verification becomes compute-bound. Falsified by a workload where higher acceptance yields monotonic throughput gains at large batch.

**H3 (KV transfer).** KV-quantization quality loss is predicted by mismatch between the method’s outlier assumptions and the attention layout: per-channel key methods will degrade more on MLA/MQA than on MHA at matched bit-width, and removing the recent-token residual window will hurt multi-step reasoning more than open-ended chat. Falsified if degradation is uniform across attention variants and residual-window ablations.

**H4 (hybrid retrieval).** A hybrid linear/full-attention model with a minority of full-attention layers will match full attention on retrieval-heavy long-context benchmarks, while a pure linear/SSM model will fail exact retrieval as context and retrieval density grow. Falsified if a pure linear/SSM model matches full attention on RULER-style exact retrieval.

**H5 (diffusion/agent mismatch).** Diffusion LMs’ single-shot throughput advantage will not transfer to multi-turn agent workloads with high prefix reuse unless block-causal or hybrid caching is added; per-turn latency will exceed an autoregressive engine with prefix caching. Falsified if a bidirectional diffusion LM without KV caching beats an autoregressive engine on agent traces.

## Bibliography

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
