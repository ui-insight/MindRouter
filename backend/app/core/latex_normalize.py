"""
LaTeX normalization for LLM output.

Pre-processes LLM response text to ensure LaTeX math expressions are
properly delimited before reaching the browser for KaTeX rendering.

Handles common LLM output issues:
- Bare \\command{...} outside delimiters (e.g. \\frac{a}{b})
- Bare \\symbol outside delimiters (e.g. \\alpha, \\int)
- Fragmented $...$ around individual symbols instead of full expressions
- Nested $ inside subscripts/superscripts (e.g. _{$\\infty$})
- Mixed bare and delimited LaTeX on display equation lines
"""

import re

# ---------------------------------------------------------------------------
# LaTeX command / symbol lists
# ---------------------------------------------------------------------------

# Commands that take one or more {…} arguments
_BRACE_CMDS = (
    r"boxed|text|textbf|textit|mathbf|mathit|mathrm|mathcal|mathbb|mathsf|"
    r"boldsymbol|operatorname|frac|dfrac|tfrac|binom|sqrt|hat|bar|vec|dot|"
    r"ddot|tilde|widetilde|widehat|overline|underline|overbrace|underbrace|"
    r"underset|overset|stackrel|cancel|bcancel|xcancel|not"
)

# Commands that need a second {…} group
_TWO_ARG_CMDS = {"frac", "dfrac", "tfrac", "binom", "underset", "overset", "stackrel"}

# Operator / symbol commands (no required braces, but may have _/^ )
_OPERATORS = (
    # Big operators
    r"lim|sum|int|iint|iiint|oint|prod|coprod|bigcup|bigcap|bigvee|bigwedge|"
    # Named functions
    r"sup|inf|max|min|log|ln|sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|"
    r"arcsin|arccos|arctan|det|gcd|deg|dim|ker|exp|arg|hom|Pr|"
    # Arrows and relations
    r"to|gets|mapsto|implies|iff|Rightarrow|Leftarrow|Leftrightarrow|"
    r"rightarrow|leftarrow|leftrightarrow|hookrightarrow|longrightarrow|"
    # Common symbols
    r"infty|pm|mp|times|div|cdot|cdots|ldots|ddots|vdots|star|circ|bullet|"
    r"oplus|otimes|dagger|Box|"
    # Relations
    r"approx|equiv|cong|sim|simeq|neq|ne|leq|le|geq|ge|ll|gg|prec|succ|"
    r"subset|supset|subseteq|supseteq|in|notin|ni|"
    # Logic and set ops
    r"forall|exists|nexists|nabla|partial|ell|Re|Im|wp|aleph|"
    r"emptyset|varnothing|neg|lnot|land|lor|cup|cap|setminus|"
    # Geometry and misc
    r"triangle|angle|perp|parallel|mid|nmid|vdash|models|therefore|because|"
    # Delimiters (when used as commands)
    r"bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|left|right|"
    # Greek lowercase
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|"
    r"tau|upsilon|phi|varphi|chi|psi|omega|"
    # Greek uppercase
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega"
)

# Compiled patterns
_BRACE_CMD_RE = re.compile(rf"\\({_BRACE_CMDS})\{{")
_OPERATOR_RE = re.compile(rf"\\({_OPERATORS})\b")
_BEGIN_END_RE = re.compile(
    r"(\\begin\{([a-zA-Z*]+)\}[\s\S]*?\\end\{\2\})"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_brace(text: str, pos: int) -> int:
    """Return index after matching '}' for '{' at *pos*, or -1."""
    if pos >= len(text) or text[pos] != "{":
        return -1
    depth = 1
    i = pos + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i if depth == 0 else -1


def _consume_sub_super(text: str, pos: int) -> int:
    """Consume _{…}, ^{…}, _x, ^x, _$…$, ^$…$ sequences starting at *pos*."""
    while pos < len(text):
        if text[pos] not in ("_", "^") or pos + 1 >= len(text):
            break
        nxt = text[pos + 1]
        if nxt == "{":
            end = _match_brace(text, pos + 1)
            if end == -1:
                break
            pos = end
        elif nxt == "$":
            dollar_end = text.find("$", pos + 2)
            if dollar_end == -1:
                break
            pos = dollar_end + 1
        else:
            pos += 2  # single char: _x ^2
    return pos


def _is_inside_dollar(text: str, pos: int) -> bool:
    """Check if *pos* is inside an inline $…$ region (simple parity check)."""
    count = 0
    i = 0
    while i < pos:
        if text[i] == "$" and (i == 0 or text[i - 1] != "\\"):
            count += 1
        i += 1
    return count % 2 == 1


def _is_inside_code_block(text: str, pos: int) -> bool:
    """Check if *pos* falls inside a fenced code block (``` ... ```)."""
    # Find all ``` positions before pos
    fence_count = 0
    i = 0
    while i < pos:
        if text[i:i + 3] == "```":
            fence_count += 1
            i += 3
        else:
            i += 1
    return fence_count % 2 == 1


# ---------------------------------------------------------------------------
# Phase 0: Strip $ from subscripts/superscripts
# ---------------------------------------------------------------------------

_SUB_DOLLAR_RE = re.compile(r"([_^])\{\$([^$]*)\$\}")


def strip_dollars_in_subscripts(text: str) -> str:
    """_{$\\infty$} -> _{\\infty}, ^{$n$} -> ^{n}."""
    prev = None
    while prev != text:
        prev = text
        text = _SUB_DOLLAR_RE.sub(r"\1{\2}", text)
    return text


# ---------------------------------------------------------------------------
# Phase 1: Detect math-heavy lines and consolidate
# ---------------------------------------------------------------------------

_PROSE_WORD_RE = re.compile(r"[a-zA-Z]{4,}")
_DOLLAR_PAIR_RE = re.compile(r"\$[^$]+\$")
_BACKSLASH_CMD_RE = re.compile(r"\\[a-zA-Z]+")


def wrap_math_lines(text: str) -> str:
    """Detect lines that are primarily math and consolidate fragmented $."""
    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        trimmed = line.strip()

        # Track code fences
        if trimmed.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue

        if not trimmed:
            result.append(line)
            continue

        # Skip already-delimited display math
        if trimmed.startswith("$$") or trimmed.startswith("\\["):
            result.append(line)
            continue

        # Count math indicators
        dollar_pairs = len(_DOLLAR_PAIR_RE.findall(trimmed))
        backslash_cmds = len(_BACKSLASH_CMD_RE.findall(trimmed))

        # Need at least 2 already-wrapped $…$ fragments to consolidate
        if dollar_pairs < 2:
            result.append(line)
            continue

        # Check for prose: strip LaTeX commands and dollar blocks, look for real words
        stripped = _BACKSLASH_CMD_RE.sub("", trimmed)
        stripped = _DOLLAR_PAIR_RE.sub(" ", stripped)
        prose_words = _PROSE_WORD_RE.findall(stripped)

        # Any prose word of 4+ letters means this is mixed content — don't consolidate
        if len(prose_words) >= 1:
            result.append(line)
            continue

        # This is a math-heavy line — strip all $ and wrap in $$
        content = trimmed.replace("$", "")
        indent = line[: len(line) - len(line.lstrip())]
        result.append(f"{indent}$${content}$$")

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Phase 2: Wrap bare \command{…} blocks
# ---------------------------------------------------------------------------


def wrap_bare_commands(text: str) -> str:
    """Find \\frac{a}{b}, \\sqrt{x}, etc. outside $ delimiters and wrap in $."""
    result = []
    i = 0
    while i < len(text):
        m = _BRACE_CMD_RE.search(text, i)
        if not m:
            result.append(text[i:])
            break

        cmd_start = m.start()
        cmd_name = m.group(1)
        brace_start = m.end() - 1  # position of '{'

        # Skip if inside existing $ or code block
        if _is_inside_dollar(text, cmd_start) or _is_inside_code_block(text, cmd_start):
            result.append(text[i : brace_start + 1])
            i = brace_start + 1
            continue

        end = _match_brace(text, brace_start)
        if end == -1:
            result.append(text[i : brace_start + 1])
            i = brace_start + 1
            continue

        # Two-arg commands: consume second {} group
        if cmd_name in _TWO_ARG_CMDS and end < len(text) and text[end] == "{":
            end2 = _match_brace(text, end)
            if end2 != -1:
                end = end2

        # Consume trailing sub/superscripts
        end = _consume_sub_super(text, end)

        # Extract and strip inner $ from wrapped content
        result.append(text[i:cmd_start])
        inner = text[cmd_start:end].replace("$", "")
        result.append(f"${inner}$")
        i = end

    return "".join(result)


# ---------------------------------------------------------------------------
# Phase 3: Wrap bare operators / symbols
# ---------------------------------------------------------------------------


def wrap_bare_operators(text: str) -> str:
    """Find bare \\alpha, \\int_{0}^{1}, etc. and wrap in $."""
    result = []
    i = 0
    while i < len(text):
        m = _OPERATOR_RE.search(text, i)
        if not m:
            result.append(text[i:])
            break

        cmd_start = m.start()

        if _is_inside_dollar(text, cmd_start) or _is_inside_code_block(text, cmd_start):
            result.append(text[i : cmd_start + len(m.group(0))])
            i = cmd_start + len(m.group(0))
            continue

        end = cmd_start + len(m.group(0))

        # For \left and \right, consume through matching delimiter + braces
        cmd_name = m.group(1)
        if cmd_name in ("left", "right", "bigl", "bigr", "Bigl", "Bigr",
                         "biggl", "biggr", "Biggl", "Biggr"):
            # These are delimiter sizing commands — consume the next char
            if end < len(text) and text[end] in ("(", ")", "[", "]", "{", "}", "|", ".", "\\"):
                end += 1

        end = _consume_sub_super(text, end)

        result.append(text[i:cmd_start])
        inner = text[cmd_start:end].replace("$", "")
        result.append(f"${inner}$")
        i = end

    return "".join(result)


# ---------------------------------------------------------------------------
# Phase 4: Extend $\func$(args) -> $\func(args)$
# ---------------------------------------------------------------------------

_EXTEND_FUNC_RE = re.compile(
    r"\$([^$]*\\[a-zA-Z]+(?:\{[^}]*\})*[^$]*)\$\s*(\([^()]*(?:\([^()]*\)[^()]*)*\))"
)


def extend_func_args(text: str) -> str:
    """$\\sin$(x) -> $\\sin(x)$, $\\widehat{f}$(\\xi) -> $\\widehat{f}(\\xi)$."""
    return _EXTEND_FUNC_RE.sub(lambda m: f"${m.group(1)}{m.group(2)}$", text)


# ---------------------------------------------------------------------------
# Phase 5: Merge adjacent $…$ blocks
# ---------------------------------------------------------------------------

_MERGE_RE = re.compile(r"\$([^$]+)\$([^\n$]{0,25}?)\$([^$]+)\$")
_PROSE_3_RE = re.compile(r"[a-zA-Z]{3,}")


def merge_adjacent_math(text: str) -> str:
    """Merge $a$ op $b$ -> $a op b$ when gap is short and math-like."""
    prev = None
    while prev != text:
        prev = text
        def _replacer(m):
            a, gap, b = m.group(1), m.group(2), m.group(3)
            # At least one side must have a LaTeX command
            if "\\" not in a and "\\" not in b:
                return m.group(0)
            # Don't merge across sentence boundaries
            if re.search(r"[.!?]", gap):
                return m.group(0)
            # Don't merge across prose words
            if _PROSE_3_RE.search(gap):
                return m.group(0)
            # Don't merge across structural braces or sub/superscripts
            if re.search(r"[{}]", gap):
                return m.group(0)
            sep = gap.strip()
            return f"${a}{' ' + sep + ' ' if sep else ' '}{b}$"
        text = _MERGE_RE.sub(_replacer, text)
    return text


# ---------------------------------------------------------------------------
# Phase 6: Wrap bare \begin{env}…\end{env}
# ---------------------------------------------------------------------------


def wrap_bare_environments(text: str) -> str:
    """Wrap \\begin{env}…\\end{env} in $$ if not already delimited."""
    def _replacer(m):
        start = m.start()
        before = text[:start]
        dollar_count = len(re.findall(r"(?<!\\)\$", before))
        if dollar_count % 2 == 1:
            return m.group(0)  # inside existing $
        trimmed = before.rstrip()
        if trimmed.endswith("$"):
            return m.group(0)  # already delimited
        return f"$${m.group(0)}$$"
    return _BEGIN_END_RE.sub(_replacer, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_latex(text: str) -> str:
    """
    Run the full LaTeX normalization pipeline on LLM output text.

    This should be called on complete response content (not partial chunks)
    before sending to the browser for rendering.
    """
    if not text or "\\" not in text:
        return text  # fast path: no LaTeX commands at all

    text = strip_dollars_in_subscripts(text)
    text = wrap_bare_commands(text)
    text = wrap_bare_operators(text)
    text = extend_func_args(text)
    text = merge_adjacent_math(text)
    # Math-line consolidation runs AFTER wrapping/merging so it sees $…$ fragments
    text = wrap_math_lines(text)
    text = wrap_bare_environments(text)

    return text
