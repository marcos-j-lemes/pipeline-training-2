#!/usr/bin/env python3
"""
wiki_scraper.py — Download and clean Wikipedia articles (Portuguese by default)

Usage:
    python wiki_scraper.py Amendoeira
    python wiki_scraper.py "Maçã" "Laranjeira" "Pinheiro"
    python wiki_scraper.py --file lista.txt
    python wiki_scraper.py --lang en Photosynthesis
    python wiki_scraper.py --output-dir ./artigos Amendoeira
"""

import argparse
import re
import sys
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

# ─── DEFAULT SETTINGS ────────────────────────────────────────────────────────

DEFAULT_LANG = "pt"           # Wikipedia language subdomain (pt, en, es, …)
DEFAULT_OUTPUT_DIR = "dataset"      # Folder where .txt files are saved
DEFAULT_ENCODING = "utf-8"    # Output file encoding

# ─── HTML CLEANER ─────────────────────────────────────────────────────────────

class WikiHTMLParser(HTMLParser):
    """
    Walks Wikipedia's HTML and extracts:
      - <h1>–<h6>  → section titles (with = markers for hierarchy)
      - <p>         → body paragraphs
      - <li>        → list items  (inside <ul>/<ol>)
      - <td>/<th>   → table cells (plain text only)

    Skips entirely:
      infoboxes, navboxes, TOC, references sidebar, edit-section links,
      <script>, <style>, <sup> (footnote numbers), <figure>/<img>.
    """

    # CSS classes that wrap noise we want to skip
    SKIP_CLASSES = {
        "infobox", "infobox_v2", "navbox", "navbox-inner",
        "toc", "tocnumber", "mw-editsection", "reference",
        "reflist", "refbegin", "thumbinner", "thumbcaption",
        "thumb", "gallery", "wikitable", "sidebar", "hatnote",
        "mw-references-wrap", "noprint", "mw-jump-link",
        "catlinks", "printfooter", "mw-footer",
    }

    # Tags whose entire subtree we always skip
    SKIP_TAGS = {"script", "style", "figure", "img", "sup",
                 "footer", "nav", "head"}

    # HTML void elements do not emit a matching end tag. If one appears inside
    # a skipped subtree, it must not increase the skip depth.
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
        self._buf: list[str] = []          # text buffer for current block
        self._skip_depth: int = 0          # >0 → inside a skipped subtree
        self._content_depth: int = 0
        self._found_content: bool = False
        self._current_tag: str = ""
        self._heading_level: int = 0
        self._in_list_item: bool = False
        self._in_table_cell: bool = False
        self._pending_newline: bool = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _flush(self):
        text = " ".join(self._buf).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            if self._heading_level:
                marker = "=" * self._heading_level
                self.lines.append(f"\n{marker} {text} {marker}")
            elif self._in_list_item:
                self.lines.append(f"  • {text}")
            elif self._in_table_cell:
                pass  # table cells are collected separately; skip here
            else:
                self.lines.append(text)
        self._buf = []

    def _classes(self, attrs):
        for name, val in attrs:
            if name == "class" and val:
                return set(val.split())
        return set()

    # ── parser callbacks ─────────────────────────────────────────────────────

    def handle_starttag(self, tag, attrs):
        classes = self._classes(attrs)
        if self._content_depth > 0 and tag not in self.VOID_TAGS:
            self._content_depth += 1
        elif "mw-parser-output" in classes:
            self._content_depth = 1
            self._found_content = True
            return

        if self._skip_depth > 0:
            if tag not in self.VOID_TAGS:
                self._skip_depth += 1
            return

        if tag in self.SKIP_TAGS:
            if tag not in self.VOID_TAGS:
                self._skip_depth = 1
            return

        if classes & self.SKIP_CLASSES:
            self._skip_depth = 1
            return

        # Check id-based skips (TOC, references, categories…)
        ids = {v for k, v in attrs if k == "id" and v}
        skip_ids = {"toc", "mw-toc-heading", "References",
                    "Notes", "catlinks", "footer", "mw-navigation"}
        if ids & skip_ids:
            self._skip_depth = 1
            return

        if self._content_depth == 0:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._heading_level = int(tag[1])
        elif tag == "p":
            self._flush()
        elif tag == "li":
            self._flush()
            self._in_list_item = True
        elif tag in ("ul", "ol"):
            self._flush()
        elif tag in ("td", "th"):
            self._flush()
            self._in_table_cell = True
        elif tag == "br":
            self._buf.append(" ")
        elif tag == "a":
            pass  # we harvest the visible text, ignore href

    def handle_startendtag(self, tag, attrs):
        # Self-closing tags such as <meta /> or <img /> do not affect nesting.
        return

    def handle_endtag(self, tag):
        if self._skip_depth > 0:
            self._skip_depth -= 1
            if self._content_depth > 0 and tag not in self.VOID_TAGS:
                self._content_depth -= 1
            return

        if self._content_depth == 0:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._heading_level = 0
        elif tag == "p":
            self._flush()
        elif tag == "li":
            self._flush()
            self._in_list_item = False
        elif tag in ("td", "th"):
            self._flush()
            self._in_table_cell = False

        if tag not in self.VOID_TAGS:
            self._content_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0 or self._content_depth == 0:
            return
        cleaned = data.replace("\n", " ").replace("\t", " ")
        if cleaned.strip():
            self._buf.append(cleaned)

    def handle_entityref(self, name):
        pass  # handled by html.parser automatically

    # ── result ────────────────────────────────────────────────────────────────

    def get_text(self) -> str:
        self._flush()
        return "\n".join(self.lines).strip()


# ─── WIKIPEDIA FETCHER ────────────────────────────────────────────────────────

def build_url(article: str, lang: str) -> str:
    encoded = urllib.parse.quote(article.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/wiki/{encoded}"


def fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"Article not found (404): {url}")
        raise RuntimeError(f"HTTP error {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e


def extract_title(html: str) -> str:
    """Pull the canonical article title from <title>…</title>."""
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        # Wikipedia: "Article – Wikipédia, a enciclopédia livre"
        title = m.group(1).split("–")[0].split("-")[0].strip()
        return title
    return "article"


def sanitize_filename(name: str) -> str:
    """Turn a title into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(". ")
    return name or "article"


# ─── MAIN LOGIC ───────────────────────────────────────────────────────────────

def scrape(article: str, lang: str, output_dir: Path, use_article_filename: bool = False) -> Path:
    url = build_url(article, lang)
    print(f"  Downloading: {url}")

    html = fetch_html(url)
    title = extract_title(html)

    parser = WikiHTMLParser()
    parser.feed(html)
    text = parser.get_text()

    if not text:
        raise RuntimeError("Parsed text is empty — page structure may have changed.")

    # Add a header with metadata
    header = (
        f"Título: {title}\n"
        f"URL: {url}\n"
        f"Idioma: {lang}\n"
        f"{'─' * 60}\n\n"
    )
    content = header + text

    filename_base = article if use_article_filename else title
    filename = sanitize_filename(filename_base) + ".txt"
    out_path = output_dir / filename
    out_path.write_text(content, encoding=DEFAULT_ENCODING)
    return out_path

def read_article_list(file_path: str) -> list[str]:
    """Read article names from a local .txt file, one article per line."""
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"File not found: {file_path}")

    articles = []
    for line in path.read_text(encoding=DEFAULT_ENCODING).splitlines():
        article = line.strip()
        if article and not article.startswith("#"):
            articles.append(article)
    return articles


def main():
    parser = argparse.ArgumentParser(
        description="Download and clean Wikipedia articles to plain .txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "articles",
        nargs="*",
        metavar="ARTICLE",
        help='Article name(s) as they appear on Wikipedia (e.g. "Amendoeira")',
    )
    parser.add_argument(
        "--file",
        "--input-file",
        dest="input_file",
        metavar="TXT",
        help="Read article names from a .txt file, one article per line",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        metavar="LANG",
        help=f"Wikipedia language code (default: {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory to save .txt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--filename",
        action="store_true",
        help="Use the article name as the output filename instead of the title"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        articles = list(args.articles)
        if args.input_file:
            articles.extend(read_article_list(args.input_file))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not articles:
        parser.error("provide at least one ARTICLE or use --file lista.txt")

    success, failure = 0, 0

    for article in articles:
        print(f"\n[+] {article}")
        try:
            out = scrape(article, args.lang, output_dir, args.filename)
            print(f"  Saved -> {out}")
            success += 1
        except (ValueError, RuntimeError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failure += 1

    print(f"\nDone. {success} saved, {failure} failed.")
    if failure:
        sys.exit(1)


if __name__ == "__main__":
    main()