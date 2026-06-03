"""Parse LaTeX source to map \includegraphics calls to figure numbers.

Usage within arxiv-figure-dl.py:
    from arxiv_tex_parser import TexFigureParser
    parser = TexFigureParser(temp_dir)
    mappings = parser.parse_all(extracted_figure_paths)
    for m in mappings:
        print(f"  Fig {m.figure_number}: {m.source_path} -> {m.resolved_path}")
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FigureIncludegraphics:
    """One \\includegraphics found inside a figure environment."""
    source_path: str            # Raw LaTeX path (e.g., 'figures/trend')
    resolved_path: str | None   # Actual file in tarball (e.g., 'figures/trend.pdf')
    figure_number: int          # Sequential figure counter
    subfigure_index: int | None = None  # a/b/c for subfigures
    caption: str = ""
    keyword: str = ""
    env_type: str = "figure"    # figure, figure*, wrapfigure


TEX_EXTENSIONS = {'.tex'}
FIGURE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.eps'}
FIGURE_LIKE_ENVS = {'figure', 'figure*', 'wrapfigure'}
STOPWORDS = frozenset({
    'the', 'a', 'an', 'of', 'in', 'for', 'and', 'with', 'to', 'is', 'on',
    'by', 'as', 'that', 'are', 'from', 'this', 'we', 'our', 'be', 'at', 'or',
    'it', 'its', 'per', 'vs', 'via', 'over', 'under', 'into', 'than',
    'was', 'were', 'been', 'has', 'have', 'had', 'but', 'not', 'no',
    'can', 'may', 'also', 'very', 'just', 'more', 'less', 'each', 'all',
    'both', 'between', 'among', 'using', 'based', 'shown', 'show', 'fig',
    'table', 'figure', 'left', 'right', 'top', 'bottom', 'middle', 'center',
})


class TexFigureParser:
    """Parse LaTeX source files to map \\includegraphics to figure numbers."""

    def __init__(self, tex_root_dir: str):
        self.tex_root = Path(tex_root_dir)

    # ----- Main entry point -----

    def parse_all(self, extracted_figure_paths: set[str]) -> list[FigureIncludegraphics]:
        """Parse all tex files and return \\includegraphics mapped to figure numbers."""
        main_tex = self._find_main_tex()
        if not main_tex:
            return []

        tex_files = self._collect_tex_files(main_tex)
        if not tex_files:
            return []

        result = []
        figure_counter = 0

        for tex_file, virtual_path in tex_files:
            try:
                raw = tex_file.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue

            text = self._strip_comments(raw)
            envs = self._find_environments(text)

            for env_start, env_end, env_name in envs:
                if env_name not in FIGURE_LIKE_ENVS:
                    continue

                env_body = text[env_start:env_end]
                figure_counter += 1

                caption = self._extract_caption(env_body)
                subfigs = self._detect_subfigures(env_body)

                if subfigs:
                    for idx, sf_paths in enumerate(subfigs):
                        for inc_path in sf_paths:
                            resolved = self._resolve_path(
                                inc_path, str(tex_file.parent),
                                extracted_figure_paths
                            )
                            result.append(FigureIncludegraphics(
                                source_path=inc_path,
                                resolved_path=resolved,
                                figure_number=figure_counter,
                                subfigure_index=idx,
                                caption=caption,
                                keyword=self._generate_keyword(caption),
                                env_type=env_name,
                            ))
                else:
                    inc_paths = self._find_includegraphics_paths(env_body)
                    multi = len(inc_paths) > 1
                    for idx, inc_path in enumerate(inc_paths):
                        resolved = self._resolve_path(
                            inc_path, str(tex_file.parent),
                            extracted_figure_paths
                        )
                        result.append(FigureIncludegraphics(
                            source_path=inc_path,
                            resolved_path=resolved,
                            figure_number=figure_counter,
                            subfigure_index=idx if multi else None,
                            caption=caption,
                            keyword=self._generate_keyword(caption),
                            env_type=env_name,
                        ))

        return result

    # ----- File discovery -----

    def _find_main_tex(self) -> Path | None:
        """Find the main .tex file (contains \\documentclass)."""
        for f in sorted(self.tex_root.rglob('*.tex')):
            try:
                content = f.read_text(encoding='utf-8', errors='replace')
                if '\\documentclass' in content:
                    return f
            except Exception:
                continue
        # Fallback: any .tex in root
        for f in sorted(self.tex_root.glob('*.tex')):
            return f
        # Fallback: any .tex at all
        for f in sorted(self.tex_root.rglob('*.tex')):
            return f
        return None

    def _collect_tex_files(self, main_tex: Path) -> list[tuple[Path, str]]:
        """Follow \\input/\\include chains to collect all tex files in order."""
        included = []
        visited = set()
        todo = [(main_tex, str(main_tex.relative_to(self.tex_root)))]

        while todo:
            tex_file, virtual_path = todo.pop(0)
            canonical = tex_file.resolve()
            if canonical in visited:
                continue
            visited.add(canonical)
            included.append((tex_file, virtual_path))

            try:
                content = tex_file.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue

            for ref in self._find_includes(content):
                resolved = self._resolve_include(ref, tex_file.parent)
                if resolved and resolved.exists() and resolved.suffix in TEX_EXTENSIONS:
                    try:
                        rel = str(resolved.relative_to(self.tex_root))
                    except ValueError:
                        rel = str(resolved)
                    todo.append((resolved, rel))

        return included

    @staticmethod
    def _find_includes(text: str) -> list[str]:
        """Find \\input{path} and \\include{path} references."""
        paths = []
        for m in re.finditer(r'\\(?:input|include)\s*(?:\{[^}]*\})', text):
            inner = m.group()[m.group().index('{'):]
            result = _extract_braces(inner, 0)
            if result is not None:
                path = result[0].strip()
                # Skip self-references and non-file includes
                if path and not path.startswith('.') and not path.startswith('!'):
                    paths.append(path)
        return paths

    @staticmethod
    def _resolve_include(ref: str, base_dir: Path) -> Path | None:
        """Resolve an \\input reference to an actual file."""
        candidates = [
            base_dir / ref,
            base_dir / (ref + '.tex'),
            Path(ref),
            Path(ref + '.tex'),
        ]
        # Also search relative to tex_root
        for c in candidates:
            if c.exists():
                return c
        return None

    # ----- Comment stripping -----

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Strip LaTeX comments. Handles \\verb and verbatim blocks."""
        lines = text.split('\n')
        result = []
        in_verbatim = False
        verbatim_envs = {'verbatim', 'lstlisting', 'minted', 'Verbatim'}

        for line in lines:
            stripped = line.lstrip()

            if in_verbatim:
                if stripped.startswith('\\end{'):
                    for env in verbatim_envs:
                        if stripped.startswith(f'\\end{{{env}'):
                            in_verbatim = False
                            break
                result.append('')
                continue

            # Check for verbatim start
            if stripped.startswith('\\begin{'):
                for env in verbatim_envs:
                    if stripped.startswith(f'\\begin{{{env}'):
                        in_verbatim = True
                        result.append('')
                        break
                else:
                    result.append(line)
                continue

            # Strip inline comments: first unescaped % to end
            i = 0
            while i < len(line):
                if line[i] == '\\' and i + 1 < len(line):
                    i += 2  # skip escaped char
                    continue
                if line[i] == '%':
                    result.append(line[:i])
                    break
                i += 1
            else:
                result.append(line)

        return '\n'.join(result)

    # ----- Environment finding -----

    @staticmethod
    def _find_environments(text: str) -> list[tuple[int, int, str]]:
        """Find all \\begin{env}...\\end{env} blocks.

        Returns list of (env_start_pos, env_end_pos, env_name).
        Environments at any nesting depth are returned.
        """
        envs = []
        stack = []  # (env_name, begin_pos)
        i = 0
        n = len(text)

        while i < n:
            # Look for \begin{ or \end{
            bs = text.find('\\begin{', i)
            es = text.find('\\end{', i)

            if bs == -1 and es == -1:
                break

            if bs != -1 and (es == -1 or bs < es):
                # \begin{ found
                result = _extract_braces(text, bs + 6)  # after \begin{
                if result:
                    env_name, end = result
                    # Check if it's commented: search backwards for % on the same line
                    line_start = text.rfind('\n', 0, bs)
                    if line_start == -1:
                        line_start = 0
                    line_before = text[line_start:bs].strip()
                    if not line_before or not line_before.startswith('%'):
                        stack.append((env_name, bs))
                    i = end
                else:
                    i = bs + 6
            else:
                # \end{ found
                result = _extract_braces(text, es + 4)  # after \end{
                if result:
                    env_name, end = result
                    if stack and stack[-1][0] == env_name:
                        _, begin_pos = stack.pop()
                        # Collect all environments regardless of nesting depth
                        envs.append((begin_pos, es, env_name))
                    i = end
                else:
                    i = es + 4

        return envs

    # ----- \includegraphics parsing -----

    @staticmethod
    def _find_includegraphics_paths(text: str) -> list[str]:
        """Find all \\includegraphics[opts]{path} in text.

        Returns list of file paths (with extension if present, without [] opts).
        """
        paths = []
        i = 0
        n = len(text)
        pattern = '\\includegraphics'

        while True:
            pos = text.find(pattern, i)
            if pos == -1:
                break
            pos += len(pattern)
            # Skip whitespace
            while pos < n and text[pos] in ' \t\r\n':
                pos += 1
            # Skip optional arguments [opts][opts]...
            while pos < n and text[pos] == '[':
                depth = 1
                pos += 1
                while pos < n and depth > 0:
                    if text[pos] == '[':
                        depth += 1
                    elif text[pos] == ']':
                        depth -= 1
                    pos += 1
                while pos < n and text[pos] in ' \t\r\n':
                    pos += 1
            # Now expect {path}
            if pos < n and text[pos] == '{':
                result = _extract_braces(text, pos)
                if result:
                    path, _ = result
                    path = path.strip()
                    # Strip double braces: {{file}} -> file
                    if path.startswith('{') and path.endswith('}'):
                        path = path[1:-1].strip()
                    if path:
                        paths.append(path)
                i = pos + 1
            else:
                i = pos + 1

        return paths

    # ----- Subfigure detection -----

    @staticmethod
    def _detect_subfigures(env_body: str) -> list[list[str]]:
        """Detect subfigure environments within a figure body.

        Returns list of subfigure groups, each containing a list of \\includegraphics paths.
        """
        # Method 1: \begin{subfigure}...\end{subfigure}
        subfigs = []
        i = 0
        n = len(env_body)

        while True:
            # Look for \begin{subfigure} or \subfigure
            sf_begin = env_body.find('\\begin{subfigure}', i)
            sf_legacy = env_body.find('\\subfigure{', i)
            sf_legacy_b = env_body.find('\\subfigure[', i)

            candidates = []
            if sf_begin != -1:
                candidates.append((sf_begin, 'modern'))
            if sf_legacy != -1:
                candidates.append((sf_legacy, 'legacy_noopt'))
            if sf_legacy_b != -1:
                candidates.append((sf_legacy_b, 'legacy_opt'))

            if not candidates:
                break

            candidates.sort()
            pos, style = candidates[0]

            if style in ('legacy_noopt', 'legacy_opt'):
                # \subfigure[caption]{content}
                # For legacy_noopt: skip \subfigure -> {
                # For legacy_opt: skip [caption] then {
                search_start = pos + len('\\subfigure')
                if style == 'legacy_opt':
                    search_start = env_body.find('{', search_start)
                    if search_start == -1:
                        break
                else:
                    # Already pointing at {
                    pass

                # Find the content braces
                brace_pos = env_body.find('{', search_start)
                if brace_pos == -1:
                    i = pos + 1
                    continue
                result = _extract_braces(env_body, brace_pos)
                if result:
                    content, _ = result
                    inc_paths = TexFigureParser._find_includegraphics_paths(content)
                    if inc_paths:
                        subfigs.append(inc_paths)
                i = brace_pos + 1

            elif style == 'modern':
                # \begin{subfigure}[pos]{width}...\end{subfigure}
                end = env_body.find('\\end{subfigure}', pos)
                if end == -1:
                    i = pos + 1
                    continue
                block = env_body[pos:end + len('\\end{subfigure}')]

                # Find subfigure content: after \begin{subfigure}[opts]{width}
                # Simpler: just extract includegraphics from the whole subfigure block
                inc_paths = TexFigureParser._find_includegraphics_paths(block)
                if inc_paths:
                    subfigs.append(inc_paths)
                i = end + len('\\end{subfigure}')

            else:
                i = pos + 1

        return subfigs

    # ----- Caption extraction -----

    @staticmethod
    def _extract_caption(env_body: str) -> str:
        """Extract \\caption{text} from a figure environment body."""
        # Find \caption (not inside nested \begin{}...\end{})
        i = 0
        n = len(env_body)

        while i < n:
            pos = env_body.find('\\caption', i)
            if pos == -1:
                break

            # Skip if it's \captionof or other variant
            after = pos + len('\\caption')
            if after < n and env_body[after:after+1] not in ('{', '['):
                i = after
                continue

            # Skip optional [short caption]
            if after < n and env_body[after] == '[':
                depth = 1
                after += 1
                while after < n and depth > 0:
                    if env_body[after] == '[':
                        depth += 1
                    elif env_body[after] == ']':
                        depth -= 1
                    after += 1
                while after < n and env_body[after] in ' \t\r\n':
                    after += 1

            # Extract {long caption}
            if after < n and env_body[after] == '{':
                result = _extract_braces(env_body, after)
                if result:
                    caption, _ = result
                    return TexFigureParser._clean_caption(caption)

            i = after + 1

        return ""

    @staticmethod
    def _clean_caption(caption: str) -> str:
        """Strip LaTeX commands and formatting from caption text."""
        # Remove \label{...}
        caption = re.sub(r'\\label\{[^}]*\}', '', caption)
        # Remove \ref{...}, \cite{...}
        caption = re.sub(r'\\(?:ref|cite|eqref|autoref|pageref)\{[^}]*\}', '', caption)
        # Remove \textbf{...}, \textit{...}, \emph{...} etc. - keep content
        caption = re.sub(r'\\(?:textbf|textit|emph|textsl|textsc|texttt|textsf|textnormal)\{', '', caption)
        # Remove math mode $...$ and $$...$$
        caption = re.sub(r'\$\$[^$]*\$\$', '', caption)
        caption = re.sub(r'\$[^$]*\$', '', caption)
        # Remove \url{...}, \href{...}{...}
        caption = re.sub(r'\\url\{[^}]*\}', '', caption)
        caption = re.sub(r'\\href\{[^}]*\}\{[^}]*\}', '', caption)
        # Remove \protect, \label, etc.
        caption = re.sub(r'\\(?:protect|label|quad|qquad|hfill|vspace|hspace)\s*', ' ', caption)
        # Remove \footnote{...}
        caption = re.sub(r'\\footnote\{[^}]*\}', '', caption)
        # Remove remaining control sequences
        caption = re.sub(r'\\[a-zA-Z]+', ' ', caption)
        # Remove braces
        caption = caption.replace('{', '').replace('}', '')
        # Collapse whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    # ----- Path resolution -----

    @staticmethod
    def _resolve_path(ref_path: str, tex_file_dir: str,
                      extracted_paths: set[str]) -> str | None:
        """Resolve a LaTeX \\includegraphics path to an extracted figure file.

        Tries multiple strategies to match the LaTeX reference to actual files.
        """
        if not ref_path:
            return None

        # Normalize the extracted paths set
        def norm(p: str) -> str:
            p = p.replace('\\', '/')
            while p.startswith('./'):
                p = p[2:]
            return p.strip()

        normalized = {norm(p) for p in extracted_paths}

        ref = norm(ref_path)

        # Strategy 1: exact match
        if ref in normalized:
            return ref

        # Strategy 2: exact match without extension
        base = Path(ref).stem
        if ref.endswith(base):  # no extension
            for ext in FIGURE_EXTENSIONS:
                candidate = ref + ext
                if candidate in normalized:
                    return candidate

        # Strategy 3: try as filename only (strip directory)
        fname = Path(ref).name
        for np in normalized:
            if Path(np).name == fname:
                return np
        fstem = Path(ref).stem
        for np in normalized:
            if Path(np).stem == fstem:
                return np

        # Strategy 4: resolve relative to tex file directory
        rel = os.path.normpath(os.path.join(tex_file_dir, ref))
        rel = norm(rel)
        if rel in normalized:
            return rel
        # without extension
        if rel.endswith(Path(rel).stem):
            for ext in FIGURE_EXTENSIONS:
                candidate = rel + ext
                if candidate in normalized:
                    return candidate

        # Strategy 5: try with common prefixes stripped
        for np in normalized:
            np_stem = Path(np).stem
            ref_stem = Path(ref).stem
            if np_stem == ref_stem:
                return np

        return None

    # ----- Keyword generation -----

    @staticmethod
    def _generate_keyword(caption: str, max_tokens: int = 3,
                          max_length: int = 40) -> str:
        """Generate a file-safe keyword from caption text."""
        if not caption:
            return "figure"

        # Lowercase and strip non-alphanumeric
        caption = caption.lower()
        # Remove parenthetical notes
        caption = re.sub(r'\([^)]*\)', '', caption)
        # Tokenize
        tokens = re.split(r'[^a-zA-Z0-9]+', caption)
        tokens = [t for t in tokens if t and t not in STOPWORDS
                  and (len(t) >= 3 or t.isdigit())]

        if not tokens:
            return "figure"

        # Take first meaningful tokens
        keyword = '_'.join(tokens[:max_tokens])
        keyword = keyword[:max_length].rstrip('_')
        return keyword if keyword else "figure"


# ----- Utility functions -----

def _extract_braces(text: str, start: int) -> tuple[str, int] | None:
    """Extract content between balanced { } starting at text[start] which should be '{'.

    Returns (content, end_position) or None if no balanced braces found.
    """
    if start >= len(text) or text[start] != '{':
        return None

    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return (text[start + 1:i], i + 1)
        elif ch == '%' and i > start:
            # Inline comment inside braces - unusual but possible
            # Only if not escaped
            if i == 0 or text[i-1] != '\\':
                return (text[start + 1:i], i + 1)
        i += 1
    return None
