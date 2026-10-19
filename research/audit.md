# Audit memo — `research/REPORT.md`

I read the report and checked 12 distinct external sources: the six 2026 citations, the ASD paper, FlashAttention-4, the roofline chapter, Orca, PagedAttention, Leviathan et al., and DeepSeek-V2. I did not run benchmarks; every performance figure below is author-reported, and I make no claim of reproduction.

**Bottom line:** I found no fabricated citations, no wrong dates, and no arithmetic errors. The KV calculation is correct. The defects are interpretive: one conceptual conflation, two one-sided engineering implications, and some imprecise wording. Nothing rises to high severity.

## Findings

### 1. Medium — ASD is a greedy verifier, but the report places it in a stochastic-sampling argument
**Type:** interpretation / source scope. **Location:** §3, paragraph after the table.

The report moves from “exact speculative sampling preserves the target distribution” to “greedy verification, exact stochastic rejection sampling and heuristic acceptance are different contracts,” then cites ASD as an example of “relaxed verification.” ASD is explicitly a **greedy** verifier: it replaces first-mismatch truncation with a budgeted longest-prefix rule using target-logit regret, a per-block exception cap, and a request-level budget. It is not a relaxation of the rejection sampler and cannot be substituted into one without re-deriving the acceptance rule. The paper itself says it “does not guarantee identical outputs, semantic preservation, or task correctness.”

Specific correction: split the sentence into two claims — “ASD relaxes *greedy* verification under a regret budget” and “relaxing exact stochastic rejection sampling is a different, distribution-level contract.” Add that ASD’s own results show >95% hash divergence on GSM8K/MATH-500 and accuracy deltas down to −1.52 pp, so it belongs in an “approximate greedy modes” category, not the exact-sampling category.

Source: https://arxiv.org/abs/2608.03447

### 2. Medium-low — the AV “trimming” warning reads as an open risk; the source treats it as a rejected design
**Type:** source fact / interpretation. **Location:** §4, paragraph after the table.

The report says: “trimming based on the probability of the sampled draft token can change the proposal distribution … This deserves a targeted correctness test before any throughput claim.” That is technically true, but the vLLM post does not ship such a trimmer. It explicitly rejects `logit(q_sampled)` and `−log(−log(q_sampled))` because the trim decision becomes correlated with the sampled token, so surviving drafts are no longer distributed as `q`; it ships `logit(q_max)`, which is independent of the sampled token. Footnote 1 gives a concrete two-token counterexample (`p=[0.5,0.5]`, `q=[0.2,0.8]`, trimming at `q(x)<0.5` yields `p_out=[0.4,0.6]`). The engineering lesson is stronger than the report’s: losslessness was a **design constraint** that eliminated candidate features, not an unresolved risk.

Specific correction: “AV’s shipped estimator uses `logit(q_max)` precisely because trimming on `q_sampled` would break the rejection sampler’s `q`; the footnote is a proof sketch, not an empirical finding. A test should verify the implementation uses a token-independent feature.”

Source: https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator

### 3. Low-medium — “Its abstract establishes the proposal” mischaracterizes a full experimental paper
**Type:** source fact. **Location:** §3, line 87.

The report says the ASD preprint’s “abstract establishes the proposal, not independent reproduction or universal deployment readiness.” The paper is not abstract-only: it has a full experimental section, seven-task primary matrix, cross-family DSpark/EAGLE3/Medusa results, ablations, appendices, and a code link. The correct limitation is not “abstract only” but “author-run experiments on one L20 GPU with vLLM 0.11.0, not independently reproduced and not shown to generalize to other targets/drafters.”

Specific correction: “The paper reports author-run experiments (Qwen3-14B + DSpark-14B, L20, vLLM 0.11.0) but no independent reproduction; its scope is greedy decoding and small-scale serving.”

Source: https://arxiv.org/abs/2608.03447

### 4. Low-medium — AV limitations are omitted from the engineering implication
**Type:** recommendation overreach. **Location:** §4 table, 14 October AV row.

The table says “Draft length/verification budget can respond to observed acceptance and measured cost instead of remaining fixed.” The blog’s limitations materially narrow this: the target attention backend must support variable-length verification, and MTP/EAGLE benefit less than DFlash/DSpark because they pay drafting cost for tokens that may be trimmed before verification. Without those caveats, the row reads as a general adaptive-verification win.

Specific correction: add “requires variable-length verification support; gains are smaller for auto-regressive drafters (MTP/EAGLE) that pay to draft tokens later trimmed.”

Source: https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator

### 5. Low — push-mode implication omits the routing tradeoff
**Type:** recommendation overreach. **Location:** §4 table, 14 October push row.

The report says “Cache transfer setup and placement can lie directly on the first-token critical path.” True, but one-sided. The blog states that push mode selects the decode instance earlier, which reduces the router’s freedom to do load-aware routing; pull mode remains the better fit for bidirectional KV transfer or deferred decode selection. The report should not present push as a strict improvement.

Specific correction: “Push moves setup off the critical path at the cost of earlier decode placement; pull retains load-aware routing and bidirectional transfer.”

Source: https://vllm.ai/blog/2026-10-14-nixl-push-connector

### 6. Low — “decode-produced state” is imprecise
**Type:** source fact / wording. **Location:** §4 table, 12 October row.

The bidirectional-KV blog says the decode node retains **both** prompt-prefix KV and generated-response KV, then returns block metadata to the prefill node. “Decode-produced state” suggests only the response KV; the reusable block set includes prefill-produced prompt KV. Correction: “decode-*retained* state (prompt + generated KV).”

Source: https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload

### 7. Low — “draft length/verification budget” conflates two quantities
**Type:** source fact / wording. **Location:** §4 table, 14 October AV row.

AV does not change how many tokens the drafter proposes. It trims the set of drafted tokens sent to the target verifier. “Draft length” is the wrong term; “verified draft length” or “verification budget” is right. Correction: “the number of drafted tokens *verified* per step responds to predicted acceptance and measured draft/verify cost.”

Source: https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator

### 8. Low — the roofline needs its companion upper bound
**Type:** mathematical derivation. **Location:** §2, lines 44–50.

The max formula is correctly described as an optimistic lower bound. The source also gives the matching upper bound: `T ≤ T_math + T_comms`, with the two bounds differing by at most 2×. Adding that sentence would stop readers treating the max as a tight estimate. The report’s caveat about dependencies and critical paths is otherwise correct.

Source: https://jax-ml.github.io/scaling-book/roofline/

### 9. Low — cite the FlashAttention-4 paper for the technical claim
**Type:** sourcing. **Location:** §4 table, 5 March row.

The blog is an author summary; the technical claim is better cited to “FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling,” arXiv 2603.05451 (5 Mar 2026). The blog’s numbers (1605 TFLOP/s, 71%, 1.3× cuDNN 9.13, 2.7× Triton) are B200 BF16 author results, not independently reproduced. Correction: cite the paper and label the numbers author-reported.

Source: https://arxiv.org/abs/2603.05451

### 10. Info — “Verified source” should mean URL/date verified, not result verified
**Type:** framing. **Location:** §4 table header and preamble.

The report already says “Blog claims remain claims by the authors; this research did not reproduce their benchmark results,” which is the right hedge. But “Verified source” plus “concrete primary-source observations” is stronger than the evidence. Rename the column “Primary source (author claim; URL/date verified)” and keep “Engineering implication” clearly marked as inference.

Sources: all six 2026 URLs above.

## Verified claims

**Source facts (URL and content checked):**
- All six 2026 citations resolve with the stated dates: FA4 blog 5 Mar 2026; SGLang DeepSeek-V4 25 Apr 2026; vLLM disaggregation guide 29 Sep 2026; bidirectional KV 12 Oct 2026; push connector 14 Oct 2026; AV estimator 14 Oct 2026.
- FA4: Blackwell asymmetric scaling; H100→B200 BF16 tensor throughput 1→2.25 PFLOPs while SFU count and shared-memory bandwidth stay flat; forward bottleneck is exp/SFU, backward is shared-memory traffic; 1605 TFLOP/s, 71%, 1.3× cuDNN 9.13, 2.7× Triton.
- SGLang DeepSeek-V4: 1.6T Pro / 284B Flash; SWA plus C4/C128 compression; ShadowRadix; three heterogeneous KV pools and two compression-state pools; ring-size doubling under speculation; 1M-token context claim.
- vLLM disaggregation guide: frontend/render tier, KV connectors, proxy/router, worker execution; the L40S example shows slow transfer can erase P/D gains (P/D TTFT 2.2 s vs 0.7 s collocated).
- Bidirectional KV: decode retains blocks for `decoder_kv_blocks_ttl` (default 480 s); `kv_recompute_threshold` default 64 tokens; 1.6× Qwen3-32B and 3.3× GLM-5-FP8 TTFT reduction at long prompts; 30–130 ms transfer cost.
- Push KV: up to 3× TTFT and 30% TPOT improvement for Qwen3-32B on 1P1D P5en/EFA; pull mode gives the router more freedom and is preferred for bidirectional transfer.
- AV: online per-position logistic regression (`K+1` parameters), IRLS refit, `logit(q_max)` feature; AUC 0.8336, log-loss 0.4284, ECE 0.0100; `q_sampled` features rejected as lossy.
- ASD: arXiv 2608.03447v3, 30 Aug 2026; greedy; 3.05–15.26% fixed-work throughput, 7.78% mean on seven Qwen3-14B + DSpark-14B tasks; >95% hash divergence; L20, vLLM 0.11.0.
- Orca: OSDI ’22, iteration-level scheduling and selective batching; 36.9× throughput at equal latency on GPT-3 175B vs FasterTransformer.
- PagedAttention: arXiv 2309.06180 (title/page verified; I did not re-read the method).
- DeepSeek-V2 MLA: caches compressed `c_KV` and decoupled RoPE key `k_R`; 15.6K vs 110.6K KV elements/token (small MoE), 34.6K vs 860.2K (large), 93.3% KV reduction.
- Roofline: max as lower bound, sum as upper bound, ≤2× gap; arithmetic-intensity framing and ~295-token critical batch on H100 BF16.

**Mathematical derivations (checked by hand):**
- `2 × 32 × 8 × 128 × 2 × 8192 = 1,073,741,824 bytes = 1 GiB`. Correct.
- 16 requests → 16 GiB; 32 KV heads → 4 GiB/request. Correct.
- 8B params × 2 bytes = 16×10⁹ bytes = 14.90 GiB. Correct.
- 8B params at 4 bits = 4×10⁹ bytes raw. Correct.
- `2MKN` FLOPs and `2 × active linear-weight parameters × tokens` are the standard multiply-add conventions. Correct, with the stated exclusions.

**Historical attributions:** the 2017 Transformer was encoder–decoder; MQA is 2019, GQA 2023; Orca/PagedAttention/SGLang/Sarathi/DistServe/Mooncake and the quantization papers are dated correctly in the table. I did not fetch primary sources for the pre-2017 rows.

## Could not verify

- The internal memos (`foundations.md`, `systems.md`, `frontiers.md`) and their “access levels”; I did not read them. The report’s claim that they contain historical primary citations is therefore unverified.
- The statement that no model-quality or engine-performance benchmark was run. Plausible, but not auditable from `REPORT.md`.
- Tensor-parallel KV-head replication. Architecturally well known (KV heads can be replicated when `H_KV < TP`), but I did not fetch a source for it in this audit.
- The systems memo’s comparison of vLLM, SGLang, TensorRT-LLM, llama.cpp, and ONNX Runtime.
- FA2/FA3 PDFs, the SGLang paper, Sarathi-Serve, DistServe, Mooncake, GPTQ, SmoothQuant, AWQ, Mamba, and LLaDA. I spot-checked only the citations above; the untested URLs may still be fine.
- The claim that ASD is representative of relaxed verification. It is one of several works it cites (MARS, Fuzzy, Cactus); the report says “explicitly explores,” not “the only one,” so this is defensible but not established.
- Any benchmark number in the six 2026 posts. I verified that the authors report them; I did not reproduce them, and the vLLM posts themselves say results depend on workload and configuration.

## Assessment

The report’s arithmetic and citation hygiene are unusually good. The build order in §5 is explicitly labeled a recommendation, and I do not think it overstates the evidence; the overreach is localized to the §4 table, where two implications are broader than the sources’ stated scope. The “main conclusion” is an engineering judgment — the sources support the importance of state management and scheduling, not the specific sequencing of “correct reference first.” That is a reasonable recommendation, but it should be read as such, not as a finding derived from the literature.
