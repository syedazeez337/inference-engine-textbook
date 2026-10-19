import fs from 'node:fs';
const root = new URL('./', import.meta.url);
function edit(name, changes, extra = '') {
  const file = new URL(name, root);
  let text = fs.readFileSync(file, 'utf8');
  for (const [prefix, replacement] of changes) {
    let found = false;
    text = text.split(/\r?\n/).map(line => {
      if (!line.startsWith(prefix)) return line;
      found = true;
      return replacement;
    }).join('\n');
    if (!found) throw new Error(`Missing paragraph ${name}: ${prefix}`);
  }
  fs.writeFileSync(file, text.trimEnd() + '\n' + extra);
}
edit('systems.md', [
  ['**Derivation.** Disaggregation is latency isolation', '**Derivation.** vLLM positions its disaggregated-prefill implementation for TTFT/ITL isolation rather than raw-throughput improvement [18]. This is not a universal verdict: DistServe and Mooncake study SLO-constrained goodput/effective capacity. Disaggregation can win or lose depending on interference, placement, transfer and load. Measure raw throughput and goodput separately at equal resource budgets.'],
], '\n## Review qualifications\n\nThe recovery agent edited the source agent\'s research; first-person reading claims refer to that original run. Benchmark figures above are author-reported historical observations, and the memo does not reproduce every workload/configuration detail needed for comparison. Do not rank engines from those figures. See [final review](review_memos.md) for missing conditions.\n\nDecode context parallelism is documented in [vLLM DCP](https://vllm.ai/blog/2026-08-07-decode-context-parallelism); CP is not limited to prefill. TensorRT-LLM has a [PyTorch-native backend](https://nvidia.github.io/TensorRT-LLM/torch.html), so compiled-engine constraints do not describe every path. llama.cpp exposes remote devices through its [RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md). Explicit vLLM CPU offload connectors are distinct from default recompute-based scheduler preemption. MLA replication depends on the sharding layout. Mooncake\'s link-rate recommendation is deployment-specific, not a universal disaggregation threshold.\n');
edit('frontiers.md', [
  ['The second axis is correctness.', 'The second axis is correctness. Correct rejection-sampled speculation and compatible exact prefix reuse preserve target semantics subject to numerical limits. Weight/KV quantization and approximate acceptance change the deployed target or sampling rule. Native sparse, linear-attention and diffusion checkpoints are alternative architectures, not automatically approximate executions of a dense autoregressive checkpoint. Their runtime can faithfully execute their own semantics; comparisons require separate task-quality evaluation.'],
  ['Exact speculation is a latency optimization,', 'Exact speculation often targets latency, but can also improve throughput in suitable regimes. Its benefit depends on acceptance, target/draft cost, verification work, batch size and scheduling. At high load, additional verification can consume scarce compute and reduce goodput; this must be measured rather than assumed for every large batch.'],
]);
edit('REPORT.md', [
  ['Research synthesis, cutoff', 'Research synthesis, cutoff **19 October 2026**. Literature research, source audits, recovery editing and synthesis were delegated to CommandCode agents using **deepseek/deepseek-v4.1-flash-fast**. The orchestrator reviewed and corrected the results. The systems and frontier sessions collected evidence but hit API errors during final output; same-model recovery agents consolidated their preserved drafts. Companion memos: [foundations](foundations.md), [systems](systems.md), [frontiers](frontiers.md). See [verification and limitations](VERIFICATION.md). This is a selective review, not an exhaustive survey or benchmark reproduction. Publication/content checks do not imply independent replication.'],
]);
console.log('Applied final editorial qualifications.');
