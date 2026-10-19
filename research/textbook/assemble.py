"""Arrange the completed, reviewed research into a reproducible book manuscript."""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent

def read(name):
    return (SOURCE / name).read_text(encoding='utf-8-sig')

def section(text, prefix):
    parts = re.split(r'(?m)^## ', text)
    for part in parts[1:]:
        title, _, body = part.partition('\n')
        if title.startswith(prefix):
            return body.strip()
    raise ValueError(f'Missing source section: {prefix}')

def normalize(text):
    text = re.sub(r'(?m)^(#{2,6})\s+\d+(?:\.\d+)*\.?\s+', r'\1 ', text)
    text = re.sub(r'(?m)^### ', '## ', text)
    text = re.sub(r'\]\((?:research/)?([A-Za-z_]+\.md)\)', r'](../\1)', text)
    for local in ('prompts/', 'run-manifest.json', 'tool-evidence.jsonl'):
        text = text.replace(f']({local})', f'](../{local})')
    return text.strip()

f, s, a, r = (read(n) for n in ('foundations.md','systems.md','frontiers.md','REPORT.md'))

# Restore inline references in the systems memo to actual reference anchors.
s = re.sub(r'\[(\d+)\](?!\()', lambda m: f'[{m[1]}](#ref-systems-{m[1]})', s)
s = re.sub(r'(?m)^\| (\d+) \|', lambda m: f'| <span id="ref-systems-{m[1]}">{m[1]}</span> |', s)

chapters = [
    ('From Neural Computation to Language Models', [section(f,'2.'),section(f,'3.'),section(f,'4.')]),
    ('The Execution Graph and the Runtime', [section(f,'5.')]),
    ('Hardware, Arithmetic Intensity, and the Roofline', [section(f,'6.'),section(r,'2.')]),
    ('Deriving Parameters, FLOPs, and State Memory', [section(f,'7.')]),
    ('Attention Kernels and Memory Traffic', [section(s,'1.')]),
    ('Continuous Batching, Paging, and Prefix Reuse', [section(s,'2.')]),
    ('Scheduling and Disaggregated Serving', [section(s,'3.')]),
    ('Parallelism, Collectives, and Expert Placement', [section(s,'4.')]),
    ('Quantization and Numerical Contracts', [section(a,'1.'),section(a,'2.')]),
    ('Speculative Decoding and Verification', [section(a,'3.')]),
    ('Alternative Models and Emerging Workloads', [section(a,'4.'),section(a,'5.')]),
    ('Existing Engines and the Current Research Landscape', [section(s,'5.'),section(s,'6.'),section(r,'3.'),section(r,'4.'),section(s,'Review qualifications')]),
    ('Designing and Building an Inference Engine', [section(r,'5.'),section(s,'7.'),section(f,'9.'),section(s,'9.')]),
    ('Measurement, Correctness, and Reproducibility', [section(s,'8.'),section(r,'6.')]),
    ('Research Directions and Falsifiable Hypotheses', [section(r,'7.'),section(a,'6.'),section(r,'8.'),section(f,'8.')]),
]

metadata = '''---
title: "Inference Engines: From First Principles to Serving Systems"
author: "DeepSeek research agents · reviewed and edited by Codex"
date: "Research cutoff: 19 October 2026"
lang: en
---

'''
preface = '''# About This Book {.unnumbered}

This textbook arranges the completed, reviewed research in this folder into a continuous study path. The underlying research and the reader implementation were delegated through CommandCode using **deepseek/deepseek-v4.1-flash-fast**. The chapters preserve primary citations and the source agents' stated access levels. The editor has applied the recorded corrections and converted mathematical notation for typesetting.

Read it in order for a first pass, or use the contents to revisit a subsystem. The goal is to connect a model's mathematical contract to its computation, state, hardware costs, scheduling, and measurement. The scope is a selective literature review through **19 October 2026**, not an exhaustive history or an independently reproduced set of benchmarks. Recommendations and future hypotheses are distinguished from established source findings.

::: {.definition}
**Reading the mathematics.** A parameter count and a hardware compute rate are different quantities. In inherited formulas, plain $P$ denotes a parameter count while $P_{\\mathrm{peak}}$ denotes a rate in FLOP/s. In the teaching appendices these are written $P_w$ and $R_{\\mathrm{peak}}$ to emphasize the distinction. Check the definitions and units before applying any equation.
:::

::: {.caution}
**Evidence and versions.** First-person statements about reading sources in the inherited material refer to the original research agents. Published speedups remain author-reported results tied to their original configurations. Implementation support is a snapshot at the research cutoff. The source inventory identifies reviewed reports, superseded drafts, audits, prompts and machine traces separately.
:::

'''

book = [metadata, preface]
teaching = BASE / 'teaching.md'
if teaching.exists() and teaching.stat().st_size:
    supplement = teaching.read_text(encoding='utf-8-sig')
    supplement = re.sub(r'(?m)^(# .+)$', r'\1 {.unnumbered}', supplement)
    supplement = supplement.replace('reports no speedup figures', 'does not report newly measured speedup figures')
    book.append(supplement)
for title, bodies in chapters:
    book.append(f'\n# {title}\n\n' + '\n\n'.join(normalize(body) for body in bodies))

exercises = BASE / 'exercises.md'
if exercises.exists() and exercises.stat().st_size:
    supplement = exercises.read_text(encoding='utf-8-sig')
    supplement = re.sub(r'(?m)^(# .+)$', r'\1 {.unnumbered}', supplement)
    supplement = supplement.replace('Adding gives about 93 ms', 'Adding gives about 86.1 ms')
    book.append('\n' + supplement)

book.append('\n# References and Source Access {.unnumbered}\n\nThe following catalogs retain the original source agents\' access notes. Duplicate works may appear in more than one research track because the inspected material and role differed. These notes describe reading, not independent replication.\n')
for title, content in [('Foundations references',f),('Systems references',s),('Algorithmic frontiers references',a)]:
    book.append(f'\n## {title}\n\n' + section(content,'Bibliography'))
book.append('\n# Verification and Editorial Record {.unnumbered}\n\n' + normalize(re.sub(r'^# .*\n', '', read('VERIFICATION.md'), count=1)))
book.append('\n## Foundation source-access limits\n\n' + section(f,'10.'))
book.append('\n## Audit documents\n\nThe original [audit](../audit.md) and [second review](../review_memos.md) are available alongside the [research manifest](../run-manifest.json) and [source-tool evidence](../tool-evidence.jsonl). Audit suggestions rejected or qualified by the verification record are not treated as accepted facts. Raw traces and duplicate drafts are cataloged in the complete source inventory below rather than repeated as textbook chapters.\n')

text = '\n\n'.join(book)
# Typeset the inherited memo's bold formulas without changing their meaning.
formulas = {
    '**P ≈ 12Ld² + Vd**': '$P \\approx 12Ld^2+Vd$',
    '**P ≈ L(2d² + 2g d²/h + m·d·d_ff) + Vd**': '$P \\approx L(2d^2+2gd^2/h+m d d_{\\mathrm{ff}})+Vd$',
    '**2P**': '$2P$', '**4Lnd**': '$4Lnd$', '**4LT²d**': '$4LT^2d$',
    '**C_train ≈ 6PD**': '$C_{\\mathrm{train}}\\approx6PD$',
    '**M_MHA = 2Lnd elements; bytes = 2Lndb.**': '$$M_{\\mathrm{MHA}}=2Lndb\\quad\\text{bytes per sequence}.$$ ',
    '**M_GQA = 2Lngd_h elements; bytes = 2Lngd_hb.**': '$$M_{\\mathrm{GQA}}=2Lngd_hb\\quad\\text{bytes per sequence}.$$ ',
    '**M_MQA = 2Lnd_h elements**': '$M_{\\mathrm{MQA}}=2Lnd_h\\quad\\text{elements per sequence}$',
    '**M_MLA = (d_c + d_R)Ln elements.**': '$$M_{\\mathrm{MLA}}=(d_c+d_R)Ln\\quad\\text{elements per sequence}.$$ ',
    '**M_state = L · d_inner · N elements**': '$M_{\\mathrm{state}}=L d_{\\mathrm{inner}}N\\quad\\text{elements}$',
}
for original, replacement in formulas.items():
    text = text.replace(original, replacement)

# Qualify old absolute wording that the final review corrected.
text = text.replace('decode is memory-bound, has query length one', 'ordinary small-batch decode is often memory-bound and has query length one')
text = text.replace('decode is bandwidth-bound, has query length one', 'ordinary small-batch decode is often bandwidth-bound and has query length one')
text = text.replace('Prefill is compute/softmax-bound', 'Prefill can be compute/softmax-bound')
text = text.replace('GPU↔CPU KV swapping was removed, so preemption is recompute-oriented', 'default preemption is recompute-oriented; explicit KV offload connectors are a separate capability, and')
(BASE / 'textbook.md').write_text(text, encoding='utf-8')
print(f'Assembled {len(chapters)} technical chapters; {len(text.split())} words.')
