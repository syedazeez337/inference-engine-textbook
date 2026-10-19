"""Build a validated, explicitly allowlisted GitHub Pages artifact."""
from collections import Counter
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
import hashlib
import json
import posixpath
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '_site'
BOOK = ROOT / 'research/textbook/index.html'
REPORTS = {
    'research/REPORT.md',
    'research/foundations.md',
    'research/systems.md',
    'research/frontiers.md',
    'research/VERIFICATION.md',
    'research/audit.md',
    'research/review_memos.md',
}


class Audit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.assets = [], [], []
        self.math = 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag in ('script', 'img', 'iframe') and attrs.get('src'):
            self.assets.append(attrs['src'])
        if tag == 'link' and attrs.get('rel') == 'stylesheet':
            self.assets.append(attrs.get('href'))
        self.math += tag == 'math'


def package():
    if not BOOK.is_file():
        raise SystemExit('Missing rendered textbook. Run python research/textbook/build.py first.')
    omitted = set()
    book = BOOK.read_text(encoding='utf-8-sig')

    def rewrite(match):
        before, raw_href, after, label = match.groups()
        href = unescape(raw_href)
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        target = posixpath.normpath(posixpath.join('research/textbook', unquote(parsed.path)))
        if target in REPORTS:
            public_href = urlunsplit(('', '', target, parsed.query, parsed.fragment))
            return f'<a{before}href="{escape(public_href, quote=True)}"{after}>{label}</a>'
        omitted.add(target)
        # Keep the provenance entry without linking private or unpublished artifacts.
        return '<span class="unpublished-source" title="Catalogued research source; not published with this edition">' + label + '</span>'

    book = re.sub(r'<a\b([^>]*?)href="([^"]+)"([^>]*)>(.*?)</a>', rewrite, book, flags=re.S)
    book = book.replace(
        'links open the original local files rather than duplicating their contents in the book.',
        'supporting reports are linked; archived traces and private execution evidence are catalogued but not published.'
    ).replace(
        'Each entry has a local link',
        'Published entries have a link'
    ).replace(
        'They are linked here but are not decompressed, read or re-hashed during the HTML build.',
        'They are retained locally and are not included in the published site.'
    ).replace(
        'Source links point to the original local files; raw machine traces are linked but not decompressed.',
        'Selected supporting reports are published; archived traces and execution evidence remain unpublished.'
    )

    audit = Audit()
    audit.feed(book)
    duplicate_ids = [key for key, count in Counter(audit.ids).items() if count > 1]
    if duplicate_ids or audit.assets or audit.math < 30:
        raise SystemExit(f'Invalid book: duplicate IDs={duplicate_ids}, runtime assets={audit.assets}, math={audit.math}')
    for href in audit.links:
        parsed = urlsplit(unescape(href))
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path and parsed.path not in REPORTS:
            raise SystemExit(f'Unexpected public file link: {href}')
        if not parsed.path and parsed.fragment and unquote(parsed.fragment) not in set(audit.ids):
            raise SystemExit(f'Broken book anchor: {href}')

    # A previous artifact is removed only after verifying its exact workspace location.
    resolved_output = OUTPUT.resolve()
    if resolved_output.parent != ROOT.resolve() or resolved_output.name != '_site' or OUTPUT.is_symlink():
        raise SystemExit('Refusing an unsafe artifact directory.')
    if OUTPUT.exists():
        shutil.rmtree(resolved_output)
    OUTPUT.mkdir()
    (OUTPUT / 'index.html').write_text(book, encoding='utf-8')
    (OUTPUT / '.nojekyll').write_text('', encoding='utf-8')
    for relative in sorted(REPORTS):
        source = ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f'Missing or unsupported supporting report: {relative}')
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    manifest = {
        'entrypoint': 'index.html',
        'mathml_expressions': audit.math,
        'published_files': ['index.html', '.nojekyll', *sorted(REPORTS), 'site-manifest.json'],
        'unpublished_source_links_removed': len(omitted),
        'book_sha256': hashlib.sha256(book.encode('utf-8')).hexdigest(),
    }
    (OUTPUT / 'site-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    actual = {path.relative_to(OUTPUT).as_posix() for path in OUTPUT.rglob('*') if path.is_file()}
    assert actual == set(manifest['published_files']), 'Artifact does not match the publication allowlist.'
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    package()
