# Study Exercises

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

# Worked Answers

1. $2MKN=2(4096)^3=1.374\times10^{11}$ FLOP. With $R_{\text{peak}}=10^{14}$ FLOP/s, $t\ge F/R_{\text{peak}}=1.374\times10^{-3}$ s, or 1.37 ms. This assumes a dense operation with no memory, launch, dependency, or scheduling cost; it is an idealized floor, not a runtime prediction.

2. $12Ld^2=12\cdot32\cdot4096^2=6.442\times10^9$ parameters; $Vd=32{,}000\cdot4096=1.311\times10^8$; so $P\approx6.57\times10^9$. GQA/MQA change head and KV dimensions; gated MLPs add projections; MoE activates only some experts; biases, untied embeddings, and vocabulary padding also break the formula. For 8 billion BF16 weights:
\[
8\times10^9\times2=1.6\times10^{10}\ \text{B}=16\ \text{GB}=14.90\ \text{GiB}.
\]
Realized memory is higher after scales, metadata, allocator slack, and workspaces.

3. $M=2\cdot32\cdot1\cdot8192\cdot8\cdot128\cdot2=1{,}073{,}741{,}824$ B. That is $2^{30}$ B $=1$ GiB $=1.074$ GB. Sixteen requests: $1.718\times10^{10}$ B $=16$ GiB $=17.18$ GB. Assumes equal lengths, no fragmentation, no metadata, and BF16 KV.

4. $D_w\approx P_wb_w$ bytes per step, independent of $B$ under the one-weight-read assumption. $F\approx2P_wB$, so $I=F/D_w=2B/b_w$ FLOP/B. The ridge is $I_{\text{ridge}}=R_{\text{peak}}/BW=10^{14}/(3.35\times10^{12})=29.9$ FLOP/B. Setting $2B/b_w=I_{\text{ridge}}$ gives $B_{\text{crit}}=b_wR_{\text{peak}}/(2BW)\approx29.9$. At $B=1$, $I=1$ FLOP/B, below the ridge; at $B=64$, $I=64$, above it. Weight-only quantization lowers $b_w$, saving bandwidth at small $B$ but shifting $B_{\text{crit}}$ downward; gains shrink once compute or KV traffic dominates.

5. Compute: $1.28\times10^{11}/10^{14}=1.28\times10^{-3}$ s. HBM: $1.6\times10^{10}/(3.35\times10^{12})=4.78\times10^{-3}$ s. Network: $8\times10^9/10^{11}=8.0\times10^{-2}$ s. The floor is $8.0\times10^{-2}$ s, or 80 ms, set by network. Adding gives about 93 ms, which assumes no overlap and is not the model’s bound; the roofline floor assumes perfect overlap. Real time can exceed 80 ms because of dependencies, synchronization, queueing, and other overheads.

6. Request A: $\lceil65/32\rceil=3$ blocks $=96$ slots. Request B: $\lceil31/32\rceil=1$ block $=32$ slots. Allocated $=128$ slots; useful $=96$; unused $=32$. Unused fraction of allocated capacity $=32/128=25\%$. Unused relative to useful $=32/96=33.3\%$. The first answers “what fraction of purchased slots is idle?”; the second answers “how much extra capacity is paid per useful token?”. Per request, A wastes $31/96=32.3\%$ of its allocation and B wastes $1/32=3.1\%$.

7. Exact speculative sampling accepts a draft token with probability $\min(1,p/q)$ and, on rejection, samples from the normalized residual $\max(0,p-q)$, so output is distributed as target $p$ under correct $p,q$ and arithmetic. Same-seed strings are not guaranteed because draft and target consume different random draws and kernel reduction orders; the distribution, not the realized sequence, is preserved. With a quantized target, “exact” means exact for $p_q$; quantization error relative to the full-precision model remains.

8. Fix model, GPU count, input/output length distributions, cache state, and arrival trace. Sweep offered request rate with open-loop arrivals, not a closed-loop client. Define SLOs such as p95 TTFT $\le2$ s and p95 TPOT $\le50$ ms. Measure achieved requests/s, output tokens/s, p50/p95/p99 TTFT and TPOT, rejects/errors, and queue delay. Maximum throughput is the highest achieved rate regardless of latency; SLO-goodput is the highest rate meeting both latency targets at the stated attainment. Throughput can rise while goodput saturates or falls as queues grow.
