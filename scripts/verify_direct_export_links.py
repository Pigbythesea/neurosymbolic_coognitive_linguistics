"""Check local review links and HTML anchors, without judging annotations."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurosym import direct_annotations as d


class ReviewLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.anchors.add(attrs['id'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])


def check(directory):
    root = Path(directory).resolve()
    pages = {}
    for path in sorted(root.rglob('*.html')):
        parser = ReviewLinks()
        parser.feed(path.read_text(encoding='utf-8'))
        pages[path] = parser
    if not pages:
        raise ValueError('No HTML review files found')
    links_by_file = {path:page.links for path,page in pages.items()}
    markdown_paths = sorted(root.rglob('*.md'))
    for path in markdown_paths:
        links_by_file[path] = re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8'))
    issues, count = [], 0
    for path, links in links_by_file.items():
        for href in links:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            count += 1
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            problem = None
            if not target.is_relative_to(root):
                problem = 'Link leaves the frozen review package'
            elif not target.is_file():
                problem = 'Linked file is missing'
            elif parsed.fragment and target.suffix == '.html':
                if target not in pages or unquote(parsed.fragment) not in pages[target].anchors:
                    problem = 'Linked HTML anchor is missing'
            if problem:
                issues.append({'page':path.relative_to(root).as_posix(), 'href':href, 'issue':problem})
    return {'check':'frozen HTML and Markdown review source links',
            'build_hash':d.read_json(root/'snapshot.json')['build_hash'],
            'html_pages_checked':len(pages), 'markdown_files_checked':len(markdown_paths),
            'local_links_checked':count,
            'issues':issues, 'passed':not issues,
            'semantic_correctness_evaluated':False, 'accepted_records_modified':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    directory = args.input or d.ROOT/d.read_json(d.OUTPUT/'latest.json')['directory']
    report = check(directory)
    d.save_json(d.ROOT/'artifacts/direct-annotation-export-links.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
