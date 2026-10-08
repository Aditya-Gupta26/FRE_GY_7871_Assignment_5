"""REPORT.md -> REPORT.pdf (markdown -> HTML -> WeasyPrint), same route as Assignment 1."""
import _bootstrap  # noqa: F401

from markdown_it import MarkdownIt
from weasyprint import HTML

from src.config import ROOT

CSS = """
@page { size: Letter; margin: 1.0cm 1.2cm 1.0cm 1.2cm;
        @bottom-center { content: counter(page); font-size: 8pt; color: #898781; } }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 8.6pt;
       line-height: 1.25; color: #0b0b0b; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h3 { font-size: 10.5pt; margin: 0 0 6pt 0; color: #52514e; font-weight: normal; }
h2 { font-size: 10.8pt; margin: 8pt 0 3pt 0; border-bottom: 0.6pt solid #c3c2b7; padding-bottom: 1pt; }
h4 { font-size: 9.8pt; margin: 7pt 0 2pt 0; }
p { margin: 0 0 4pt 0; text-align: justify; }
ul, ol { margin: 0 0 4pt 0; padding-left: 13pt; }
li > ul { margin: 1pt 0 1pt 0; }
li p { margin: 0; }
li { margin-bottom: 0.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 7.8pt; margin: 3pt 0 6pt 0; }
th { border-bottom: 0.8pt solid #52514e; text-align: left; padding: 1.5pt 4pt; }
td { border-bottom: 0.4pt solid #e1e0d9; padding: 1.2pt 4pt; }
img { display: block; margin: 2pt auto 0 auto; max-width: 100%; max-height: 1.8in; }
em.cap, p.cap { font-size: 8pt; color: #52514e; }
hr { border: none; border-top: 0.6pt solid #c3c2b7; margin: 6pt 0; }
code { font-size: 8.4pt; }
.refs { font-size: 7.5pt; line-height: 1.2; color: #52514e; }
"""

md = (ROOT / "REPORT.md").read_text()
# CommonMark: lists may follow a paragraph line directly and nest with 2 spaces
# (Python-Markdown needed blank lines + 4-space indents and ran the bullets together)
html = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)
page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
(ROOT / "outputs" / "REPORT.html").write_text(page)
HTML(string=page, base_url=str(ROOT)).write_pdf(ROOT / "REPORT.pdf")
print("wrote REPORT.pdf")
