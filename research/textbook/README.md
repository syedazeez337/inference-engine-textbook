# Inference Engines — HTML textbook

This folder contains an offline, self-contained HTML edition of the research in this project. `index.html` embeds the book, stylesheet and reader script in one file; it makes no network requests.

## Open the book

Open `research/textbook/index.html` directly in a browser. Double-clicking the file works, and so does a local server if you prefer one. No build step is needed to read the book.

Everything needed for reading is embedded. The provenance appendix also links to the original local research files with paths relative to `index.html`, so those links work while the book stays in this folder. If you move `index.html` by itself, the book still reads correctly but its local source links will no longer resolve.

## Math and browser support

Equations are native MathML produced by Pandoc, not images, web fonts or CDN scripts. They render offline in current Chromium-based browsers (Chrome, Edge, Brave, Opera) and Firefox. Safari has MathML support, though some complex constructs may look different. Older Chromium versions before MathML Core support require a current browser. No MathJax or KaTeX download is used.

The stylesheet requests Cambria Math, STIX Two Math or another installed math font, then falls back to the browser's MathML font. Missing math fonts do not break the text; they may change spacing.

## Rebuild

To regenerate the manuscript from the reviewed reports and the delegated teaching supplements, run `python research/textbook/assemble.py` first. This creates 15 technical chapters plus the study guide, glossary, exercises, answers, references and editorial record. It treats the recovery/synthesis drafts as historical inputs and applies the documented corrections. The two attempted long-form author runs failed to save or return a manuscript; the delivered edition assembles the completed reviewed research instead. The shorter teaching and exercise supplements were successfully returned by the requested DeepSeek model. The renderer reached its turn limit after writing its files; the editor corrected and verified the embedding and source classification before delivery.

From the project root:

```text
python research/textbook/build.py
```

Requirements:

- Python 3.9 or newer, standard library only.
- `pandoc` on `PATH`. The builder runs Pandoc with native MathML and fails with a clear message if Pandoc or `textbook.md` is missing.

The builder reads `research/textbook/textbook.md`, `research/textbook/source-inventory.json`, `template.html`, `style.css` and `reader.js`. It writes `research/textbook/index.html` and `research/textbook/build-info.json`. It does not modify the textbook or the source research files.

`build-info.json` records the research cutoff, metadata key count, chapter and inventory counts, anchor warnings, the Pandoc version and SHA-256 hashes of the textbook, inventory, output HTML and build files.

## Source inventory and provenance

The book is an editorial synthesis of the research folder with a cutoff of **19 October 2026**. It is not new empirical work, and no benchmark or model-quality evaluation was reproduced. Corrections in `VERIFICATION.md` and `review_memos.md` take precedence over earlier drafts and audit suggestions.

The generated appendix, **Complete source inventory**, lists every entry in `source-inventory.json` with one editorial role:

- canonical source
- reviewed audit
- superseded draft
- raw trace
- prompt
- reproducibility script
- manifest evidence

Each entry has a local link computed from `index.html`, plus its size. Canonical reports are also listed separately as original-source links. Multi-gigabyte NDJSON archives are linked but are never decompressed, read or re-hashed by the build.

## Printing

Use the **Print** button in the header, or the browser's print command (`Ctrl+P`, `Cmd+P`). The print stylesheet:

- hides the header controls, sidebar, reading progress and skip link;
- switches to a white background and full-width content;
- avoids page breaks immediately after headings and inside callout boxes and table rows;
- repeats table headers on new pages and allows wide tables to use the full page width;
- keeps equations and table content visible rather than clipping them.

For a very wide table, landscape orientation is the most reliable choice. A4 and Letter paper both work with the default margins. Background graphics are not required.

## Reader controls and accessibility

The book is fully readable with JavaScript disabled. The controls are progressive enhancements:

- **Dark / Light** toggles the reader theme and remembers it.
- **A− / A / A+** changes the text size and remembers it.
- **Print** opens the browser print dialog.
- The top progress bar shows reading position.
- On narrow screens, **Contents** collapses or expands the table of contents. The TOC is keyboard operable and the current section is marked with `aria-current`.
- The **Skip to main content** link is the first focusable element.
- MathML is accessible to screen readers that support it; equations are not replaced by images or code blocks.

Preferences use `localStorage` inside `try/catch`. If storage is unavailable (some `file://` configurations), the book still works with the default theme and text size for that session. No analytics, cookies, remote assets or network calls are used.

## Files

- `index.html` — generated, self-contained book.
- `textbook.md` — authored source (owned by the authoring step).
- `source-inventory.json` — complete source map.
- `build.py` — Pandoc builder and inventory appendix generator.
- `template.html`, `style.css`, `reader.js` — build inputs embedded into the output.
- `build-info.json` — build metadata and hashes.

## Troubleshooting

- **`pandoc was not found on PATH`** — install Pandoc or add its executable to `PATH`, then rerun the builder.
- **`textbook.md was not found`** — run the builder after the textbook author has written the file.
- **Equations show as boxes or spaced strangely** — update the browser and, on Windows, ensure Cambria Math is available. The MathML itself is present in the HTML source.
- **Local source links stop working** — keep `index.html` inside `research/textbook/` so its relative links to the research files remain valid.
- **Theme or text size is not remembered** — the browser may be blocking storage for local files; the session default still applies.
