from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

BASE_DIR = Path(__file__).resolve().parent
STYLE_PLACEHOLDER = "<!--INLINE_STYLE-->"
SCRIPT_PLACEHOLDER = "<!--INLINE_SCRIPT-->"
INVENTORY_PLACEHOLDER = "<!--SOURCE_INVENTORY-->"
BUILD_STAMP_PLACEHOLDER = "<!--BUILD_STAMP-->"
DEFAULT_TITLE = "Inference Engines: A Research Textbook"
RESEARCH_CUTOFF = "19 October 2026"
PANDOC_FORMAT = "markdown+tex_math_dollars+tex_math_single_backslash+fenced_divs+pipe_tables+autolink_bare_uris"

ROLE_ORDER = [
    "canonical-source",
    "reviewed-audit",
    "superseded-draft",
    "raw-trace",
    "prompt",
    "reproducibility-script",
    "manifest-evidence",
]

ROLE_LABELS = {
    "canonical-source": "Canonical source",
    "reviewed-audit": "Reviewed audit",
    "superseded-draft": "Superseded draft",
    "raw-trace": "Raw trace",
    "prompt": "Prompt",
    "reproducibility-script": "Reproducibility script",
    "manifest-evidence": "Manifest evidence",
}

ROLE_CLASSES = {
    "canonical-source": "role-canonical",
    "reviewed-audit": "role-audit",
    "superseded-draft": "role-superseded",
    "raw-trace": "role-raw",
    "prompt": "role-prompt",
    "reproducibility-script": "role-script",
    "manifest-evidence": "role-evidence",
}

ROLE_DESCRIPTIONS = {
    "canonical-source": "Edited research chapters and reports used by the book",
    "reviewed-audit": "Audit, verification and review memos; corrections take precedence",
    "superseded-draft": "Earlier machine drafts retained for provenance only",
    "raw-trace": "Final and partial machine output, compressed streams and stderr",
    "prompt": "Delegation prompts used to commission the research",
    "reproducibility-script": "Local collection, archiving and finalisation scripts",
    "manifest-evidence": "Run metadata and tool-call evidence",
}

CANONICAL_REPORTS = [
    ("research/REPORT.md", "Consolidated research report"),
    ("research/foundations.md", "Foundations: history, operators, hardware and cost accounting"),
    ("research/systems.md", "Serving systems: state, kernels, batching, scheduling and parallelism"),
    ("research/frontiers.md", "Frontiers: compression, speculation and alternative workloads"),
    ("research/VERIFICATION.md", "Verification notes and correction disposition"),
    ("research/review_memos.md", "Independent review memos and editorial corrections"),
]

BOX_TITLES = {
    "example": "Worked example",
    "definition": "Definition",
    "exercise": "Exercise",
    "takeaway": "Key takeaway",
    "caution": "Caution",
}


def fail(message: str) -> None:
    raise SystemExit(f"build.py: {message}")


def find_project_root() -> Path:
    for candidate in [BASE_DIR, *BASE_DIR.parents]:
        if (candidate / "research" / "textbook").is_dir() and (candidate / "research").is_dir():
            return candidate
    return BASE_DIR.parent.parent


PROJECT_ROOT = find_project_root()


def find_textbook() -> Path:
    candidates = [
        BASE_DIR / "textbook.md",
        PROJECT_ROOT / "research" / "textbook.md",
        PROJECT_ROOT / "textbook.md",
    ]
    seen: set[Path] = set()
    tried: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    fail("textbook.md was not found. The builder must run after the textbook author writes it. Tried:\n  " + "\n  ".join(tried))


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"required file is missing: {path}")
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        fail(f"could not read {path}: {exc}")


def write_text(path: Path, text: str) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as exc:
        fail(f"could not write {path}: {exc}")


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_front_matter(markdown: str) -> tuple[dict[str, object], int]:
    text = markdown.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            end = index
            break
    if end is None:
        return {}, 0

    metadata: dict[str, object] = {}
    authors: list[str] = []
    current_key = ""
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$", line)
        if match:
            current_key = match.group(1)
            value = match.group(2).strip()
            if current_key == "author" and not value:
                metadata[current_key] = authors
            elif value:
                metadata[current_key] = strip_quotes(value)
            else:
                metadata[current_key] = ""
            continue
        if current_key == "author":
            list_match = re.match(r"^\s*-\s+(.*)$", line)
            if list_match:
                authors.append(strip_quotes(list_match.group(1)))
    if authors:
        metadata["author"] = authors
    return metadata, len(metadata)


def pandoc_executable() -> str:
    executable = shutil.which("pandoc")
    if not executable:
        fail("pandoc was not found on PATH. Install Pandoc or make it available on PATH, then rerun the builder.")
    return executable


def pandoc_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return f"unavailable ({exc})"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.splitlines()[0].strip()
    return "unknown"


def role_for(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in {"research/synthesis.md", "research/systems_recovery.md", "research/frontiers_recovery.md"}:
        return "superseded-draft"
    if normalized.startswith("research/prompts/"):
        return "prompt"
    if normalized in {"research/run-manifest.json", "research/tool-evidence.jsonl"}:
        return "manifest-evidence"
    if normalized.endswith(".mjs") or normalized.startswith("research/scripts/"):
        return "reproducibility-script"
    if normalized in {"research/audit.md", "research/VERIFICATION.md", "research/review_memos.md"}:
        return "reviewed-audit"
    if normalized.startswith("research/raw/"):
        if normalized.endswith(".partial.md") or normalized.endswith("orchestrator-draft.md"):
            return "superseded-draft"
        return "raw-trace"
    if normalized.endswith(".md"):
        return "canonical-source"
    return "raw-trace"


def format_bytes(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "—"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} B"
    return f"{size:.1f} {unit}"


def local_link(target: Path, start: Path) -> str:
    try:
        relative = os.path.relpath(str(target), str(start))
    except ValueError:
        return target.as_uri()
    return quote(Path(relative).as_posix(), safe="/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory_html(inventory_path: Path) -> tuple[str, dict[str, object]]:
    text = read_text(inventory_path)
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"source-inventory.json is not valid JSON: {exc}")
    if not isinstance(entries, list):
        fail("source-inventory.json must contain a JSON array of file entries")

    normalized_entries: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            fail(f"source-inventory.json entry {index} is missing a string 'path'")
        path = entry["path"].replace("\\", "/")
        target = Path(path)
        if not target.is_absolute():
            target = PROJECT_ROOT / path
        normalized_entries.append(
            {
                "path": path,
                "target": target,
                "bytes": entry.get("bytes"),
                "role": role_for(path),
            }
        )

    normalized_entries.sort(key=lambda item: (ROLE_ORDER.index(str(item["role"])), str(item["path"]).lower()))
    role_counts = Counter(str(item["role"]) for item in normalized_entries)

    rows: list[str] = []
    for entry in normalized_entries:
        path = str(entry["path"])
        target = entry["target"]
        assert isinstance(target, Path)
        role = str(entry["role"])
        label = ROLE_LABELS[role]
        role_class = ROLE_CLASSES[role]
        exists = target.is_file()
        href = local_link(target, BASE_DIR)
        title_parts = [f"Local source: {path}"]
        if path.endswith(".gz"):
            title_parts.append("Compressed raw trace; linked only, not decompressed.")
        if not exists:
            title_parts.append("Not found at build time.")
        link_class = "source-link" if exists else "source-link link-missing"
        link = (
            f'<a class="{link_class}" href="{html_lib.escape(href, quote=True)}" '
            f'title="{html_lib.escape(" ".join(title_parts), quote=True)}">{html_lib.escape(path)}</a>'
        )
        rows.append(
            "<tr>"
            f'<td class="source-file">{link}</td>'
            f'<td><span class="role-badge {role_class}">{html_lib.escape(label)}</span></td>'
            f'<td class="source-size">{format_bytes(entry["bytes"])}</td>'
            "</tr>"
        )

    canonical_items: list[str] = []
    for path, description in CANONICAL_REPORTS:
        target = PROJECT_ROOT / path
        href = local_link(target, BASE_DIR)
        link_class = "source-link" if target.is_file() else "source-link link-missing"
        canonical_items.append(
            "<li>"
            f'<a class="{link_class}" href="{html_lib.escape(href, quote=True)}">{html_lib.escape(path)}</a>'
            f'<span class="canonical-description"> — {html_lib.escape(description)}</span>'
            "</li>"
        )

    legend_items = []
    for role in ROLE_ORDER:
        legend_items.append(
            '<span class="legend-item">'
            f'<span class="role-badge {ROLE_CLASSES[role]}">{html_lib.escape(ROLE_LABELS[role])}</span>'
            f'<span class="legend-description">{html_lib.escape(ROLE_DESCRIPTIONS[role])}</span>'
            "</span>"
        )

    count_bits = ", ".join(
        f"{role_counts.get(role, 0)} {ROLE_LABELS[role].lower()}" for role in ROLE_ORDER if role_counts.get(role, 0)
    )
    appendix = (
        '<section id="src-inventory-appendix" class="appendix source-appendix" aria-labelledby="src-inventory-heading">'
        '<header class="appendix-header">'
        '<p class="appendix-kicker">Provenance appendix</p>'
        '<h1 id="src-inventory-heading">Complete source inventory</h1>'
        "</header>"
        '<p class="lead">Every file in <code>source-inventory.json</code> is listed below with its editorial role, '
        "size and a local link relative to this HTML file. The inventory is the authoritative map of the research "
        "folder; links open the original local files rather than duplicating their contents in the book.</p>"
        '<h2 id="src-canonical-reports">Canonical reports</h2>'
        '<p>These reports and memos are the original sources behind the book. Corrections in the verification and '
        "review memos take precedence over earlier drafts and audit suggestions.</p>"
        '<ul class="canonical-source-links">'
        + "".join(canonical_items)
        + "</ul>"
        '<h2 id="src-inventory-table">File inventory</h2>'
        f'<p>{len(normalized_entries)} entries: {html_lib.escape(count_bits)}.</p>'
        '<div class="role-legend" role="group" aria-label="Editorial role legend">'
        + "".join(legend_items)
        + "</div>"
        '<table class="source-table">'
        "<caption>Complete inventory of research files, with editorial role and local link.</caption>"
        "<thead><tr><th scope=\"col\">Source file</th><th scope=\"col\">Role</th><th scope=\"col\">Size</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
        '<p class="source-footnote">Raw NDJSON streams and compressed archives are retained as machine evidence. '
        "They are linked here but are not decompressed, read or re-hashed during the HTML build.</p>"
        "</section>"
    )
    stats = {
        "entries": len(normalized_entries),
        "roles": dict(role_counts),
        "missing": sum(1 for item in normalized_entries if not item["target"].is_file()),
    }
    return appendix, stats


def add_box_titles(text: str) -> str:
    for class_name, title in BOX_TITLES.items():
        pattern = re.compile(r'<div\b([^>]*\bclass="[^"]*\b' + re.escape(class_name) + r'\b[^"]*"[^>]*)>', re.IGNORECASE)

        def replace(match: re.Match[str], label: str = title) -> str:
            opening = match.group(0)
            if "data-box-title" in opening:
                return opening
            following = text[match.end() : match.end() + 240].lstrip()
            if re.match(r"<h[1-6]\b", following, re.IGNORECASE):
                return opening
            return opening[:-1] + f' data-box-title="{html_lib.escape(label, quote=True)}">'

        text = pattern.sub(replace, text)
    return text


def unwrap_display_math_paragraphs(text: str) -> str:
    pattern = re.compile(r'<p>\s*(<math\b[^>]*\bdisplay="block"[^>]*>.*?</math>)\s*</p>', re.DOTALL)
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1", text)
    return text


def wrap_display_math(text: str) -> str:
    pattern = re.compile(r'(<math\b[^>]*\bdisplay="block"[^>]*>.*?</math>)', re.DOTALL)
    return pattern.sub(lambda match: '<div class="math-display" tabindex="0">' + match.group(1) + "</div>", text)


def wrap_tables(text: str) -> str:
    counter = [0]

    def replace(match: re.Match[str]) -> str:
        counter[0] += 1
        table = match.group(0)
        caption_match = re.search(r"<caption[^>]*>(.*?)</caption>", table, re.DOTALL | re.IGNORECASE)
        if caption_match:
            caption = re.sub(r"<[^>]+>", "", caption_match.group(1))
            caption = html_lib.unescape(caption).strip()
            label = f"Table: {caption[:120]}"
        else:
            label = f"Data table {counter[0]}"
        return (
            f'<div class="table-wrap" role="region" aria-label="{html_lib.escape(label, quote=True)}" tabindex="0">'
            + table
            + "</div>"
        )

    return re.sub(r"<table\b.*?</table>", replace, text, flags=re.DOTALL | re.IGNORECASE)


def anchor_report(text: str) -> dict[str, list[str]]:
    ids = re.findall(r'\bid="([^"]+)"', text)
    counts = Counter(ids)
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    hrefs = re.findall(r'href="#([^"]+)"', text)
    missing = sorted({unquote(value) for value in hrefs if value and unquote(value) not in counts})
    return {"duplicateIds": duplicates, "missingTargets": missing}


def build_stamp() -> str:
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    return f"Built {today} as a self-contained HTML file with Pandoc native MathML."


def main() -> None:
    textbook_path = find_textbook()
    inventory_path = BASE_DIR / "source-inventory.json"
    if not inventory_path.is_file():
        fail(f"source-inventory.json is missing: {inventory_path}")

    template_path = BASE_DIR / "template.html"
    style_path = BASE_DIR / "style.css"
    script_path = BASE_DIR / "reader.js"
    for required in [template_path, style_path, script_path]:
        if not required.is_file():
            fail(f"required build file is missing: {required}")

    template = read_text(template_path)
    for placeholder in [STYLE_PLACEHOLDER, SCRIPT_PLACEHOLDER, INVENTORY_PLACEHOLDER, BUILD_STAMP_PLACEHOLDER]:
        if placeholder not in template:
            fail(f"template.html is missing the placeholder {placeholder}")

    markdown = read_text(textbook_path)
    if not markdown.strip():
        fail(f"{textbook_path} is empty. Wait for the textbook author to finish, then rerun the builder.")
    metadata, metadata_count = parse_front_matter(markdown)
    executable = pandoc_executable()
    version = pandoc_version(executable)

    command = [
        executable,
        "-f",
        PANDOC_FORMAT,
        "-t",
        "html5",
        "--standalone",
        "--mathml",
        "--section-divs",
        "--number-sections",
        "--toc",
        "--toc-depth=2",
        "--wrap=none",
        "--template",
        str(template_path),
    ]
    if not metadata.get("lang"):
        command.extend(["-M", "lang=en"])
    if not metadata.get("title"):
        command.extend(["-M", f"title={DEFAULT_TITLE}"])

    try:
        result = subprocess.run(
            command,
            input=markdown,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        fail(f"could not run pandoc: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        fail(f"pandoc failed with exit code {result.returncode}:\n{detail}")

    body_html = result.stdout
    chapter_count = len(re.findall(r'<section\b[^>]*\bclass="[^"]*\blevel1\b[^"]*"', body_html))
    inventory_html, inventory_stats = build_inventory_html(inventory_path)

    body_html = body_html.replace(INVENTORY_PLACEHOLDER, inventory_html)
    body_html = add_box_titles(body_html)
    body_html = unwrap_display_math_paragraphs(body_html)
    body_html = wrap_display_math(body_html)
    body_html = wrap_tables(body_html)
    body_html = body_html.replace(BUILD_STAMP_PLACEHOLDER, build_stamp())

    style = read_text(style_path)
    script = read_text(script_path)
    body_html = body_html.replace(STYLE_PLACEHOLDER, "<style>\n" + style + "\n</style>")
    body_html = body_html.replace(SCRIPT_PLACEHOLDER, "<script>\n" + script + "\n</script>")

    for placeholder in [STYLE_PLACEHOLDER, SCRIPT_PLACEHOLDER, INVENTORY_PLACEHOLDER, BUILD_STAMP_PLACEHOLDER]:
        if placeholder in body_html:
            fail(f"unreplaced template placeholder remains: {placeholder}")

    anchors = anchor_report(body_html)
    output_path = BASE_DIR / "index.html"
    write_text(output_path, body_html)

    hash_paths = [
        textbook_path,
        inventory_path,
        output_path,
        BASE_DIR / "build.py",
        template_path,
        style_path,
        script_path,
    ]
    readme_path = BASE_DIR / "README.md"
    if readme_path.is_file():
        hash_paths.append(readme_path)
    hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in hash_paths
    }

    info_title = str(metadata.get("title") or DEFAULT_TITLE)
    info = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generator": "research/textbook/build.py",
        "pandocVersion": version,
        "pandocFormat": PANDOC_FORMAT,
        "researchCutoff": RESEARCH_CUTOFF,
        "bookTitle": info_title,
        "metadataCount": metadata_count,
        "metadata": {
            "title": info_title,
            "author": metadata.get("author", []),
            "date": metadata.get("date", ""),
            "lang": metadata.get("lang", "en"),
        },
        "counts": {
            "chapters": chapter_count,
            "sourceInventoryEntries": inventory_stats["entries"],
            "missingInventoryFiles": inventory_stats["missing"],
            "roles": inventory_stats["roles"],
        },
        "anchors": anchors,
        "hashes": hashes,
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    info_path = BASE_DIR / "build-info.json"
    write_text(info_path, json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    print(f"Built {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  chapters: {chapter_count}; inventory entries: {inventory_stats['entries']}; metadata keys: {metadata_count}")
    print(f"  pandoc: {version}")
    if anchors["duplicateIds"]:
        print("  warning: duplicate IDs: " + ", ".join(anchors["duplicateIds"]), file=sys.stderr)
    if anchors["missingTargets"]:
        print("  warning: internal links without targets: " + ", ".join(anchors["missingTargets"]), file=sys.stderr)


if __name__ == "__main__":
    main()
