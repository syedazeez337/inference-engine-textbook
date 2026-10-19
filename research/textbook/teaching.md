# How to Study This Book

This is a selective, research-based study book about the forward execution of trained neural models, especially decoder-only Transformers. The literature cutoff is 19 October 2026. It is not an exhaustive survey and not a benchmark reproduction: it teaches mechanisms and derivations, reports no speedup figures, and treats every published result as a hypothesis to test on your own hardware. Source quality varies; flag claims resting only on abstracts or metadata. Distinguish source reports, cost-model derivations, and design recommendations.

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

# Notation and Units

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

# Glossary

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
