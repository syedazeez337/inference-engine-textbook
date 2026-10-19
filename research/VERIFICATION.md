# Verification and research provenance

Cutoff: 19 October 2026. The user requested delegation through **CommandCode with `deepseek/deepseek-v4.1-flash-fast` and web search**.

## What actually ran

The installed CLI identified itself as CommandCode v1.84.0 and listed the exact requested model. A short probe successfully invoked `web_search` and `web_fetch` for the original FlashAttention paper. Three independent literature sessions were then launched for foundations, systems, and algorithmic frontiers. A separate session audited the synthesis. A systems recovery/editor session and a final synthesis session were delegated to the same model.

All successful invocations used read-only plan permissions, disabled skill discovery and automatic updates, and used `--no-session`. Web search and fetch were available by default; `--tools-enable` is unnecessary for them because that flag only restores tools withheld in headless mode. No alternate model was substituted. Exact per-run model IDs, status, tool counts and provider-reported usage are in [run-manifest.json](run-manifest.json). Tool requests, responses and errors are in [tool-evidence.jsonl](tool-evidence.jsonl). Prompts are in [prompts](prompts/).

The orchestrator initially did independent source checks and drafted a synthesis. After the user clarified that substantive work should stay delegated, further literature synthesis and recovery were assigned to CommandCode. The orchestrator's subsequent role was review, corrections and packaging.

## Evidence levels

- **Primary-source content inspected:** supports the specific method or implementation description read. It does not establish independent replication.
- **Abstract/metadata inspected:** supports title, date, authors and claims actually present in the abstract; it does not imply the full paper was read.
- **Author/project blog or documentation:** supports what its authors report or document, subject to its hardware, workload and version limitations.
- **Derived:** formulas and illustrative arithmetic, with explicit assumptions.
- **Recommended or predicted:** engineering judgments and testable hypotheses, not established experimental results.

The agent bibliographies state their source access. These are agent-reported reading levels, with source-tool evidence retained for inspection. The orchestrator spot-checked central sources; it did not independently reread every cited paper. Neither the agents nor orchestrator reproduced engine benchmarks or trained/evaluated a model in this task. This is a selective review rather than a systematic survey with exhaustive inclusion criteria.

## Audit disposition

The separate [audit memo](audit.md) found no arithmetic errors in the illustrative KV/weight-memory calculations and no fabricated citations among its checked sources. That statement applies only to the audited subset.

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

The [second review](review_memos.md) checked overstated runtime comparisons and formula assumptions. Editorial corrections include restricting the 12Ld² parameter estimate to its architecture assumptions; identifying batch weight-traffic amortization correctly; qualifying incomplete prefill traffic estimates and recurrent-state accounting; acknowledging llama.cpp RPC, TensorRT-LLM's PyTorch backend and decode context parallelism; and distinguishing default preemption from explicit offload. The synthesis's missing `d` in `4LBnd` and its `log(q_max)`/`logit(q_max)` transcription error were also corrected. The original returned memos remain in raw evidence.

An attempted recovery launch with `--effort medium` was rejected because this CLI supports `off`, `low`, `high`, and `max`. A subsequent `--effort low` attempt tried to persist global configuration and failed under filesystem permissions. Recovery proceeded with the original working command and no effort override; no permission bypass or global configuration change was made.

Future hypotheses are proposed experiments. Hardware selection, exact checkpoint support, quality targets and latency requirements remain design inputs to choose before implementing the engine.
