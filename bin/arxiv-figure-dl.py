#!/usr/bin/env python3
"""Download original figures from arXiv e-print source tarballs.

Usage:
    python3 bin/arxiv-figure-dl.py <arxiv_id> --outdir <output_dir> [--dpi 600] [--abbr <abbr>]

Basic mode (legacy naming):
    python3 bin/arxiv-figure-dl.py 2203.00555 --outdir wiki/images/deepnet/

Auto-mapping mode (--abbr enables LaTeX figure-number parsing):
    python3 bin/arxiv-figure-dl.py 2203.00555 --abbr deepnet --outdir wiki/images/deepnet/ --manifest

In auto-mapping mode, the script:
  1. Downloads the arXiv LaTeX source tarball
  2. Parses all .tex files to find \includegraphics inside figure environments
  3. Maps source filenames to figure numbers and caption-derived keywords
  4. Outputs auto-named files: {abbr}_fig{N}_{keyword}.png
  5. Prints a JSON manifest of all source-to-target mappings
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Load the TexFigureParser (co-located in bin/)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from arxiv_tex_parser import TexFigureParser


def download_arxiv_source(arxiv_id: str) -> bytes:
    """Download the arXiv e-print source tarball."""
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
    skip_patterns = (
        '.aux', '.bbl', '.blg', '.log', '.out', '.toc',
        'main.', 'ms.', 'paper.', 'sample.', 'sn-',
    )
    stem = Path(name).stem.lower()
    if ext == '.pdf' and any(stem.startswith(p) for p in skip_patterns):
        if not any(kw in stem for kw in ('fig', 'result', 'plot', 'model', 'arch', 'overview')):
            return False
    if ext == '.pdf' and stem in ('neurips', 'icml', 'iclr', 'aaai', 'acl', 'emnlp', 'arxiv'):
        return False
    return True


def extract_all_files(tarball_data: bytes, temp_dir: str) -> tuple[list[dict], set[str]]:
    """Extract all files from tarball. Returns (figures_list, tex_file_paths)."""
    figures = []
    tex_files = set()
    buf = io.BytesIO(tarball_data)
    with tarfile.open(fileobj=buf, mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            tar.extract(member, path=temp_dir)
            rel_path = member.name
            ext = Path(rel_path).suffix.lower()
            if ext == '.tex':
                tex_files.add(rel_path)
            elif is_figure_file(rel_path):
                figures.append({
                    'name': Path(rel_path).name,
                    'path': os.path.join(temp_dir, rel_path),
                    'ext': ext,
                    'orig_path': rel_path,
                })
    return figures, tex_files


def render_pdf_to_png(pdf_path: str, dpi: int = 600) -> list[tuple[int, bytes, int, int]]:
    """Render a PDF to PNG page images at given DPI."""
    try:
        import fitz
    except ImportError:
        print("    ERROR: PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return []

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        pages.append((page_num, pix.tobytes("png"), pix.width, pix.height))
    doc.close()
    return pages


def render_eps_to_png(eps_path: str, dpi: int = 600) -> list[tuple[int, bytes, int, int]] | None:
    """Convert EPS to PNG using Pillow (requires Ghostscript)."""
    try:
        from PIL import Image
        img = Image.open(eps_path)
        w = int(img.size[0] * dpi / 72)
        h = int(img.size[1] * dpi / 72)
        img = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return [(0, buf.getvalue(), w, h)]
    except Exception as e:
        print(f"    EPS conversion failed: {e}")
        return None


def _save_png(out_path: Path, page_data: tuple, dry_run: bool, label: str):
    """Save a PNG page to disk and print status."""
    pg_num, data, w, h = page_data
    if dry_run:
        print(f"  ✓ {label} ({w}x{h})")
        return
    with open(out_path, 'wb') as f:
        f.write(data)
    size = out_path.stat().st_size // 1024
    print(f"  ✓ {label} ({w}x{h}, {size}KB)")


def _copy_raster(src: str, dst: Path, dry_run: bool, label: str):
    """Copy a raster image file and print status."""
    size_kb = os.path.getsize(src) // 1024
    if dry_run:
        print(f"  ✓ {label} ({size_kb}KB)")
        return
    shutil.copy2(src, dst)
    try:
        from PIL import Image
        with Image.open(dst) as img:
            print(f"  ✓ {label} ({img.size[0]}x{img.size[1]}, {size_kb}KB)")
    except ImportError:
        print(f"  ✓ {label} ({size_kb}KB)")


# ----- Legacy processing (filename-based) -----

def _process_legacy(figures: list[dict], out_path: Path, dpi: int, dry_run: bool):
    """Process figures using source filenames (legacy mode, no --abbr)."""
    for fig in sorted(figures, key=lambda x: x['name']):
        fig_name = fig['name']
        ext = fig['ext']
        src_path = fig['path']
        stem = Path(fig_name).stem
        stem = re.sub(r'[^\w\-]', '_', stem)
        stem = re.sub(r'_+', '_', stem).strip('_')

        if ext == '.pdf':
            pages = render_pdf_to_png(src_path, dpi)
            if not pages:
                print(f"  ⚠ {fig_name}: empty PDF, skipping")
                continue
            if len(pages) == 1:
                out_file = out_path / f"{stem}.png"
                _save_png(out_file, pages[0], dry_run, f"{fig_name} → {out_file.name}")
            else:
                print(f"  ! {fig_name}: {len(pages)} pages")
                for pg in pages:
                    out_file = out_path / f"{stem}_p{pg[0]+1}.png"
                    _save_png(out_file, pg, dry_run, f"{fig_name} pg{pg[0]+1} → {out_file.name}")

        elif ext in ('.png', '.jpg', '.jpeg'):
            out_file = out_path / fig_name
            _copy_raster(src_path, out_file, dry_run, f"{fig_name} → {fig_name}")

        elif ext == '.eps':
            pages = render_eps_to_png(src_path, dpi)
            if not pages:
                print(f"  ⚠ {fig_name}: EPS conversion failed")
                continue
            out_file = out_path / f"{stem}.png"
            _save_png(out_file, pages[0], dry_run, f"{fig_name} → {out_file.name}")


# ----- Auto-mapping processing -----

def _process_with_mapping(figures: list[dict], tex_paths: set[str],
                           out_path: Path, dpi: int, dry_run: bool,
                           abbr: str, manifest: bool, arxiv_id: str) -> list[dict]:
    """Process figures using LaTeX auto figure-number mapping."""
    # Build set of extracted file paths (relative to temp dir)
    # Need the temp dir root - figures paths are absolute, we need relative
    extracted_paths = {f['orig_path'] for f in figures}

    # Parse LaTeX
    # We need the temp directory root. Figures[0]['path'] is <tmp>/<orig_path>
    tmp_root = os.path.commonprefix([f['path'] for f in figures]) if figures else ""
    # commonprefix works on paths character by character, so we need to find
    # the actual tmp root. It's the dir part of path - orig_path
    if figures:
        tmp_root = figures[0]['path'][:-len(figures[0]['orig_path'])]
        if tmp_root.endswith('/'):
            tmp_root = tmp_root[:-1]

    parser = TexFigureParser(tmp_root)
    mappings = parser.parse_all(extracted_paths)

    if not mappings:
        print("  Warning: no \\includegraphics references found in LaTeX source.")
        print("  Falling back to filename-based extraction.")
        _process_legacy(figures, out_path, dpi, dry_run)
        return []

    # Find unused figure files
    referenced = {m.resolved_path for m in mappings if m.resolved_path}
    unreferenced = [f for f in figures if f['orig_path'] not in referenced]

    # Build output mapping
    output_entries = []
    for m in mappings:
        if not m.resolved_path:
            print(f"  ⚠ {m.source_path}: unresolved path, skipping")
            continue

        # Find the corresponding figure info
        fig_info = next(
            (f for f in figures if f['orig_path'] == m.resolved_path),
            None
        )
        if not fig_info:
            print(f"  ⚠ {m.source_path}: resolved to {m.resolved_path} but not in extracted files")
            continue

        # Build target filename
        suffix = chr(ord('a') + m.subfigure_index) if m.subfigure_index is not None else ''
        fig_type = 'fig' if m.env_type in ('figure', 'figure*', 'wrapfigure') else 'table'
        keyword = m.keyword if m.keyword else "figure"
        target_name = f"{abbr}_{fig_type}{m.figure_number}{suffix}_{keyword}.png"

        output_entries.append({
            'source_path': m.source_path,
            'resolved_path': m.resolved_path,
            'target_name': target_name,
            'figure_number': m.figure_number,
            'subfigure_index': m.subfigure_index,
            'caption': m.caption,
            'keyword': m.keyword,
            'ext': fig_info['ext'],
            'src_abs_path': fig_info['path'],
        })

    # Render/save each mapped figure
    for entry in output_entries:
        _render_mapped(entry, out_path, dpi, dry_run)

    # Print manifest
    if manifest:
        manifest_data = {
            'arxiv_id': arxiv_id,
            'abbr': abbr,
            'outdir': str(out_path.resolve()),
            'mappings': [
                {k: v for k, v in e.items() if k != 'src_abs_path'}
                for e in output_entries
            ],
            'unreferenced': [
                {'name': f['name'], 'orig_path': f['orig_path'],
                 'size_kb': os.path.getsize(f['path']) // 1024}
                for f in unreferenced
            ],
        }
        print("\n--- MANIFEST ---")
        print(json.dumps(manifest_data, indent=2, ensure_ascii=False))

    return output_entries


def _render_mapped(entry: dict, out_path: Path, dpi: int, dry_run: bool):
    """Render/save a single mapped figure."""
    src_path = entry['src_abs_path']
    target_name = entry['target_name']
    ext = entry['ext']
    out_file = out_path / target_name

    if ext == '.pdf':
        pages = render_pdf_to_png(src_path, dpi)
        if not pages:
            print(f"  ⚠ {entry['source_path']}: empty PDF, skipping")
            return
        # For multi-page: only use first page by default
        _save_png(out_file, pages[0], dry_run,
                  f"{entry['resolved_path']} → {target_name}")

    elif ext in ('.png', '.jpg', '.jpeg'):
        _copy_raster(src_path, out_file, dry_run,
                     f"{entry['resolved_path']} → {target_name}")

    elif ext == '.eps':
        pages = render_eps_to_png(src_path, dpi)
        if not pages:
            print(f"  ⚠ {entry['source_path']}: EPS conversion failed")
            return
        _save_png(out_file, pages[0], dry_run,
                  f"{entry['resolved_path']} → {target_name}")


# ----- Main -----

def process_paper(arxiv_id: str, outdir: str, dpi: int = 600,
                  dry_run: bool = False, abbr: str = None,
                  manifest: bool = False):
    """Main processing function."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    tarball = download_arxiv_source(arxiv_id)

    with tempfile.TemporaryDirectory(prefix='arxiv_fig_') as tmp:
        figures, tex_paths = extract_all_files(tarball, tmp)
        if not figures:
            print("  No figure files found in source tarball!")
            return

        print(f"  Found {len(figures)} figure files, {len(tex_paths)} .tex files")

        if abbr:
            _process_with_mapping(figures, tex_paths, out_path, dpi,
                                  dry_run, abbr, manifest, arxiv_id)
        else:
            _process_legacy(figures, out_path, dpi, dry_run)

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
    parser.add_argument('--dry-run', action='store_true', help='List figures without processing')
    parser.add_argument('--abbr', type=str, default=None,
                        help='Paper abbreviation. Enables auto figure-number mapping from LaTeX source.')
    parser.add_argument('--manifest', action='store_true',
                        help='Print mapping manifest as JSON (implies --abbr)')

    args = parser.parse_args()

    # --manifest implies --abbr
    if args.manifest and not args.abbr:
        print("Error: --manifest requires --abbr")
        sys.exit(1)

    process_paper(args.arxiv_id, args.outdir, args.dpi,
                  args.dry_run, args.abbr, args.manifest)


if __name__ == '__main__':
    main()
