#!/usr/bin/env python3
"""Batch re-download arXiv figures for all ingested papers with auto figure-number mapping.

Usage:
    # Dry-run: validate all mappings without writing
    python3 bin/arxiv-figure-backfill.py --dry-run

    # Full run: download and process all papers
    python3 bin/arxiv-figure-backfill.py

    # Single paper
    python3 bin/arxiv-figure-backfill.py --paper deepnet
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "bin" / "arxiv-figure-dl.py"

# (arxiv_id, source_slug, image_dir, abbr)
PAPER_MAP = [
    ("1706.03762", "attention-is-all-you-need", "transformer", "transformer"),
    ("1802.05365", "deep-contextualized-word-representations", "elmo", "elmo"),
    ("1810.04805", "bert-pre-training-of-deep-bidirectional-transformers", "bert", "bert"),
    ("1811.06965", "gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism", "gpipe", "gpipe"),
    ("1905.05950", "bert-rediscovers-the-classical-nlp-pipeline", "bert-pipeline", "bert-pipeline"),
    ("1909.08053", "megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism", "megatron-lm", "megatron"),
    ("2007.14062", "big-bird-transformers-for-longer-sequences", "big-bird", "bigbird"),
    ("2104.04473", "efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm", "megatron-cluster", "meg_cluster"),
    ("2104.05343", "an-efficient-2d-method-for-training-super-large-deep-learning-models", "optimus", "optimus"),
    ("2104.09864", "roformer-enhanced-transformer-with-rotary-position-embedding", "rope", "rope"),
    ("2203.00555", "deepnet-scaling-transformers-to-1000-layers", "deepnet", "deepnet"),
    ("2205.05198", "reducing-activation-recomputation-in-large-transformer-models", "activation", "activation"),
    ("2208.07339", "llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale", "llm.int8", "llmint8"),
    ("2210.17323", "gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers", "gptq", "gptq"),
    ("2211.10438", "smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models", "smoothquant", "smoothquant"),
    ("2306.00978", "awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration", "awq", "awq"),
    ("2306.15595", "extending-context-window-via-positional-interpolation", "positional-interpolation", "pi"),
    ("2309.06180", "efficient-memory-management-for-large-language-model-serving-with-pagedattention", "pagedattention", "pagedattention"),
    ("2404.19737", "better-and-faster-large-language-models-via-multi-token-prediction", "multi-token-prediction", "multitoken"),
    ("2409.15355", "block-attention-for-efficient-prefilling", "block-attention", "blockattn"),
]


def get_existing_images(img_dir: str) -> dict[str, int]:
    """Return {filename: filesize_bytes} for existing images."""
    d = ROOT / "wiki" / "images" / img_dir
    if not d.exists():
        return {}
    result = {}
    for f in sorted(d.iterdir()):
        if f.suffix in ('.png', '.jpg', '.jpeg'):
            result[f.name] = f.stat().st_size
    return result


def run_paper(arxiv_id: str, abbr: str, outdir: str,
              dry_run: bool = False) -> dict:
    """Run arxiv-figure-dl on a single paper. Returns result dict."""
    cmd = [
        sys.executable, str(SCRIPT), arxiv_id,
        "--abbr", abbr,
        "--outdir", outdir,
        "--manifest",
        "--dry-run" if dry_run else "--no-dry-run",
    ]
    # Clean up: --no-dry-run isn't a valid flag, we just omit --dry-run
    if not dry_run:
        cmd = [c for c in cmd if c != "--no-dry-run"]

    print(f"\n{'='*70}")
    print(f"  {abbr} ({arxiv_id})")
    print(f"  Output: {outdir}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start

    # Extract manifest from stdout
    manifest = None
    if "--- MANIFEST ---" in result.stdout:
        manifest_str = result.stdout.split("--- MANIFEST ---")[1].strip()
        try:
            manifest = json.loads(manifest_str)
        except json.JSONDecodeError:
            pass

    return {
        "arxiv_id": arxiv_id,
        "abbr": abbr,
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "manifest": manifest,
        "elapsed_s": round(elapsed),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch backfill arXiv figures")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without writing files")
    parser.add_argument("--paper", type=str, default=None,
                        help="Process only one paper (by abbr)")
    args = parser.parse_args()

    papers = PAPER_MAP
    if args.paper:
        papers = [p for p in PAPER_MAP if p[3] == args.paper]
        if not papers:
            print(f"Unknown paper: {args.paper}")
            sys.exit(1)

    results = []
    failed = 0

    for i, (arxiv_id, slug, img_dir, abbr) in enumerate(papers, 1):
        outdir = str(ROOT / "wiki" / "images" / img_dir)
        old_images = get_existing_images(img_dir)
        print(f"\n[{i}/{len(papers)}] Processing {abbr}...")

        result = run_paper(arxiv_id, abbr, outdir, args.dry_run)

        if not result["success"]:
            print(f"  FAILED (rc={result['returncode']})")
            if result["stderr"]:
                print(f"  stderr: {result['stderr'][:300]}")
            failed += 1
        else:
            print(result["stdout"][:500])
            if result["manifest"]:
                m = result["manifest"]
                print(f"  → {len(m['mappings'])} figures mapped, "
                      f"{len(m['unreferenced'])} unused files skipped")

        results.append(result)

        # Rate limit: pause between papers
        if i < len(papers) and not args.dry_run:
            time.sleep(3)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary: {len(results)-failed}/{len(results)} succeeded")
    if failed:
        print(f"  Failed: {failed}")

    # In full-run mode, print comparison with old images
    if not args.dry_run and not args.paper:
        print(f"\n  Image comparison (new vs old):")
        for result in results:
            if result["manifest"]:
                abbr = result["abbr"]
                old = get_existing_images(
                    [p[2] for p in PAPER_MAP if p[3] == abbr][0]
                )
                new_count = len(result["manifest"]["mappings"])
                old_count = len(old)
                changed = "✓" if new_count > 0 else "✗"
                print(f"  [{changed}] {abbr}: {old_count} old → {new_count} new")

    return 0 if not failed else 1


if __name__ == "__main__":
    main()
