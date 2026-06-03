#!/usr/bin/env python3
"""Download original figures from arXiv e-print source tarballs.

Usage:
    python3 bin/arxiv-figure-dl.py <arxiv_id> --outdir <output_dir> [--dpi 600]

Extracts figures from the arXiv source LaTeX tarball:
  - PDF/EPS → rendered to PNG at --dpi (default 600)
  - PNG/JPG/JPEG → copied as-is (original author quality)

Example:
    python3 bin/arxiv-figure-dl.py 2203.00555 --outdir wiki/images/deepnet/
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def download_arxiv_source(arxiv_id: str) -> bytes:
    """Download the arXiv e-print source tarball."""
    # Strip version suffix if present (e.g., "2306.15595v2" → "2306.15595")
    base_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"https://arxiv.org/e-print/{base_id}"
    print(f"  Downloading {url}...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data) // 1024} KB")
    return data


def is_figure_file(name: str) -> bool:
    """Check if a filename looks like a figure."""
    ext = Path(name).suffix.lower()
    if ext not in ('.pdf', '.png', '.jpg', '.jpeg', '.eps'):
        return False
    # Skip common non-figure files
    skip_patterns = (
        '.aux', '.bbl', '.blg', '.log', '.out', '.toc',
        'main.', 'ms.', 'paper.', 'sample.', 'sn-',
    )
    stem = Path(name).stem.lower()
    if ext == '.pdf' and any(stem.startswith(p) for p in skip_patterns):
        # Check: might still be a figure if it contains keywords
        if not any(kw in stem for kw in ('fig', 'result', 'plot', 'model', 'arch', 'overview')):
            return False
    # Exclude common style/source files that happen to be PDF
    if ext == '.pdf' and stem in ('neurips', 'icml', 'iclr', 'aaai', 'acl', 'emnlp', 'arxiv'):
        return False
    return True


def extract_figures(tarball_data: bytes, temp_dir: str) -> list[dict]:
    """Extract figure files from tarball, return list of {name, path, ext}."""
    figures = []
    buf = io.BytesIO(tarball_data)
    with tarfile.open(fileobj=buf, mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not is_figure_file(member.name):
                continue
            # Extract
            tar.extract(member, path=temp_dir)
            ext = Path(member.name).suffix.lower()
            name = Path(member.name).name
            src = os.path.join(temp_dir, member.name)
            figures.append({
                'name': name,
                'path': src,
                'ext': ext,
                'orig_path': member.name,
            })
    return figures


def render_pdf_to_png(pdf_path: str, dpi: int = 600) -> bytes:
    """Render a PDF page to a PNG byte array at given DPI."""
    try:
        import fitz
    except ImportError:
        print("    ERROR: PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return None

    # Render all pages, but keep them separate (return first page only for single-page figs)
    # Multi-page PDFs are handled by the caller
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use page's bounding box to tightly crop
        pix = page.get_pixmap(dpi=dpi)
        pages.append((page_num, pix.tobytes("png"), pix.width, pix.height))
    doc.close()
    return pages


def render_eps_to_png(eps_path: str, dpi: int = 600) -> bytes | None:
    """Convert EPS to PNG using Pillow (requires Ghostscript)."""
    try:
        from PIL import Image
        img = Image.open(eps_path)
        # Scale to target DPI
        w = int(img.size[0] * dpi / 72)
        h = int(img.size[1] * dpi / 72)
        img = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return [(0, buf.getvalue(), w, h)]
    except Exception as e:
        print(f"    EPS conversion failed: {e}")
        return None


def process_paper(arxiv_id: str, outdir: str, dpi: int = 600, dry_run: bool = False):
    """Main processing function."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Download
    tarball = download_arxiv_source(arxiv_id)

    # Extract to temp
    with tempfile.TemporaryDirectory(prefix='arxiv_fig_') as tmp:
        figures = extract_figures(tarball, tmp)
        if not figures:
            print("  No figure files found in source tarball!")
            return

        print(f"  Found {len(figures)} figure files")
        for fig in sorted(figures, key=lambda x: x['name']):
            fig_name = fig['name']
            ext = fig['ext']
            src_path = fig['path']
            size_kb = os.path.getsize(src_path) // 1024

            # Determine output name
            stem = Path(fig_name).stem
            # Sanitize: replace non-alphanumeric chars (except - and _) with _
            stem = re.sub(r'[^\w\-]', '_', stem)
            stem = re.sub(r'_+', '_', stem).strip('_')

            if ext == '.pdf':
                # Render at high DPI
                pages = render_pdf_to_png(src_path, dpi)
                if pages is None:
                    print(f"  ⚠ {fig_name}: empty PDF, skipping")
                    continue

                if len(pages) == 1:
                    out_file = out_path / f"{stem}.png"
                    # Avoid overwriting old-style named files — check if fig name matches
                    if dry_run:
                        print(f"  ✓ {fig_name} → {out_file.name} ({pages[0][2]}x{pages[0][3]})")
                        continue
                    with open(out_file, 'wb') as f:
                        f.write(pages[0][1])
                    new_size = os.path.getsize(out_file) // 1024
                    print(f"  ✓ {fig_name} → {out_file.name}  ({pages[0][2]}x{pages[0][3]}, {new_size}KB, {size_kb}KB source)")
                else:
                    print(f"  ! {fig_name}: {len(pages)} pages (multi-page PDF, saving individual pages)")
                    for pg_num, pg_data, w, h in pages:
                        out_file = out_path / f"{stem}_p{pg_num+1}.png"
                        if dry_run:
                            print(f"    → {out_file.name} ({w}x{h})")
                            continue
                        with open(out_file, 'wb') as f:
                            f.write(pg_data)
                        new_size = os.path.getsize(out_file) // 1024
                        print(f"    ✓ pg{pg_num+1} → {out_file.name}  ({w}x{h}, {new_size}KB)")

            elif ext in ('.png', '.jpg', '.jpeg'):
                if dry_run:
                    print(f"  ✓ {fig_name} → {fig_name} ({size_kb}KB, raster)")
                    continue
                # Copy as-is
                out_file = out_path / fig_name
                shutil.copy2(src_path, out_file)
                w, h = 0, 0
                try:
                    from PIL import Image
                    with Image.open(out_file) as img:
                        w, h = img.size
                except ImportError:
                    pass
                dims = f" ({w}x{h}, {size_kb}KB)" if w else f" ({size_kb}KB)"
                print(f"  ✓ {fig_name} → {fig_name}{dims}")

            elif ext == '.eps':
                pages = render_eps_to_png(src_path, dpi)
                if pages is None:
                    print(f"  ⚠ {fig_name}: EPS conversion failed")
                    continue
                out_file = out_path / f"{stem}.png"
                if dry_run:
                    print(f"  ✓ {fig_name} → {out_file.name} ({pages[0][2]}x{pages[0][3]})")
                    continue
                with open(out_file, 'wb') as f:
                    f.write(pages[0][1])
                print(f"  ✓ {fig_name} → {out_file.name}  ({pages[0][2]}x{pages[0][3]})")

    # Cleanup
    print(f"  Done. Output: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description='Download original figures from arXiv e-print source',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('arxiv_id', help='arXiv ID (e.g., 2203.00555)')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for images')
    parser.add_argument('--dpi', type=int, default=600, help='DPI for PDF/EPS rendering (default: 600)')
    parser.add_argument('--dry-run', action='store_true', help='List figures without downloading')
    args = parser.parse_args()

    process_paper(args.arxiv_id, args.outdir, args.dpi, args.dry_run)


if __name__ == '__main__':
    main()
