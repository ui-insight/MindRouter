"""
Tests for backend.app.core.latex_normalize.

Validates that the LaTeX normalization pipeline:
1. Wraps bare LaTeX commands/symbols in $ or $$ delimiters
2. Preserves already-delimited math blocks ($$, \\[, \\()
3. Does not corrupt mixed content (prose + math)
4. Promotes standalone equation lines to display math
5. Handles realistic multi-line LLM output correctly

Added in v2.4.2 after discovering that _is_inside_dollar() used simple
$-parity which breaks on $$ (two $ chars = even count = "outside math"),
causing normalize_latex to corrupt every display equation.
"""

import pytest

from backend.app.core.latex_normalize import (
    normalize_latex,
    strip_dollars_in_subscripts,
    wrap_bare_commands,
    wrap_bare_operators,
    merge_adjacent_math,
    wrap_math_lines,
    wrap_bare_environments,
)


# -----------------------------------------------------------------------
# Critical regression: $$ blocks must not be corrupted (v2.4.2)
# -----------------------------------------------------------------------

class TestDisplayMathPreservation:
    """Ensure $$...$$ and other display delimiters are never mangled."""

    def test_double_dollar_preserved(self):
        text = r"$$\int_0^1 x \, dx = \frac{1}{2}$$"
        assert normalize_latex(text) == text

    def test_double_dollar_multiline_preserved(self):
        text = "$$\n\\int_0^1 x \\, dx\n$$"
        assert normalize_latex(text) == text

    def test_bracket_delimiters_preserved(self):
        text = r"\[\int_0^1 x \, dx = \frac{1}{2}\]"
        assert normalize_latex(text) == text

    def test_paren_delimiters_preserved(self):
        text = r"\(\alpha + \beta\)"
        assert normalize_latex(text) == text

    def test_no_triple_dollar_injection(self):
        """The original bug: $$ content got re-wrapped producing $$$."""
        text = r"$$\frac{a}{b}$$"
        result = normalize_latex(text)
        assert "$$$" not in result
        assert result == text

    def test_complex_display_equation_preserved(self):
        text = (
            r"$$V = 2\pi \cdot \left(1 - \frac{\sqrt{2}}{2}\right) "
            r"\cdot \frac{R^3}{3}$$"
        )
        result = normalize_latex(text)
        assert result == text

    def test_multiple_display_blocks_preserved(self):
        text = (
            "First:\n"
            r"$$\int_0^R \rho^2 \, d\rho = \frac{R^3}{3}$$"
            "\n\nSecond:\n"
            r"$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$"
        )
        result = normalize_latex(text)
        assert result.count("$$") == 4  # two open + two close
        assert "$$$" not in result


# -----------------------------------------------------------------------
# Inline $...$ preservation
# -----------------------------------------------------------------------

class TestInlineMathPreservation:
    """Existing inline $...$ should not be corrupted."""

    def test_simple_inline_preserved(self):
        text = r"The value $\alpha$ is small."
        assert normalize_latex(text) == text

    def test_inline_with_commands_preserved(self):
        text = r"We know $\frac{a}{b} + c$ is useful."
        assert normalize_latex(text) == text

    def test_multiple_inline_preserved(self):
        text = r"Given $x > 0$ and $y < 1$, we have $x + y < 2$."
        assert normalize_latex(text) == text


# -----------------------------------------------------------------------
# Bare LaTeX wrapping
# -----------------------------------------------------------------------

class TestBareLatexWrapping:
    """Bare (undelimited) LaTeX commands must be wrapped in $."""

    def test_bare_frac(self):
        # Standalone bare command on its own line gets promoted to display $$
        result = normalize_latex(r"\frac{a}{b}")
        assert r"\frac{a}{b}" in result
        assert "$" in result

    def test_bare_sqrt(self):
        result = normalize_latex(r"\sqrt{x}")
        assert r"\sqrt{x}" in result
        assert "$" in result

    def test_bare_greek(self):
        result = normalize_latex(r"\alpha")
        assert r"\alpha" in result
        assert "$" in result

    def test_bare_command_in_prose_stays_inline(self):
        """Bare command embedded in prose should be inline $, not display $$."""
        result = normalize_latex(r"The value \alpha is small.")
        assert r"$\alpha$" in result
        assert "$$" not in result

    def test_bare_integral_with_subscript(self):
        """v2.4.1 fix: \\b failed to match \\int_ because _ is a word char."""
        result = normalize_latex(r"\int_{0}^{R}")
        assert "$" in result
        assert r"\int_{0}^{R}" in result

    def test_bare_sum_with_limits(self):
        result = normalize_latex(r"\sum_{i=1}^{n}")
        assert "$" in result
        assert r"\sum_{i=1}^{n}" in result

    def test_bare_operator_before_subscript(self):
        """Operators followed by _ must be detected ((?![a-zA-Z]) fix)."""
        result = normalize_latex(r"\prod_{k=1}^{N}")
        assert "$" in result


# -----------------------------------------------------------------------
# Display math promotion
# -----------------------------------------------------------------------

class TestDisplayMathPromotion:
    """Standalone equation lines should become $$ display math."""

    def test_standalone_equation_promoted(self):
        text = r"\int_0^R \rho^2 \, d\rho = \frac{R^3}{3}"
        result = normalize_latex(text)
        assert result.startswith("$$")
        assert result.endswith("$$")

    def test_prose_line_not_promoted(self):
        text = r"The variable $\alpha$ represents distance."
        result = normalize_latex(text)
        assert not result.startswith("$$")

    def test_short_inline_in_prose_not_promoted(self):
        text = r"Substituting $\frac{a}{b}$ into the equation gives the result."
        result = normalize_latex(text)
        assert "$$" not in result


# -----------------------------------------------------------------------
# Mixed content (the real-world case)
# -----------------------------------------------------------------------

class TestMixedContent:
    """Realistic LLM output mixing prose, inline math, and display math."""

    def test_prose_with_delimited_display(self):
        text = (
            "The volume is:\n"
            r"$$\iiint_E xyz \, dV = \int_0^3 \int_0^2 \int_0^1 xyz \, dx \, dy \, dz$$"
            "\n\n**Evaluation:** Since the limits are constants."
        )
        result = normalize_latex(text)
        # Display block must be preserved exactly
        assert r"$$\iiint_E xyz \, dV" in result
        # Prose must not be wrapped
        assert "**Evaluation:**" in result

    def test_prose_with_bare_display_equation(self):
        text = (
            "Evaluating the integral:\n"
            r"\int_0^R \rho^2 \, d\rho = \frac{R^3}{3}"
        )
        result = normalize_latex(text)
        # Bare equation line should be wrapped in $$
        assert "$$" in result
        # Prose line should not be touched
        assert result.startswith("Evaluating the integral:")

    def test_realistic_multiline_output(self):
        text = (
            "The volume of the cone is:\n"
            r"$$V = \int_{0}^{2\pi} \int_{0}^{\pi/4} \int_{0}^{R} "
            r"\rho^2 \sin\phi \, d\rho \, d\phi \, d\theta$$"
            "\n\nEvaluating the $\\rho$ integral first:\n"
            r"\int_{0}^{R} \rho^2 \, d\rho = \frac{R^3}{3}"
            "\n\nThen the $\\phi$ integral:\n"
            r"\int_{0}^{\pi/4} \sin\phi \, d\phi = 1 - \frac{\sqrt{2}}{2}"
        )
        result = normalize_latex(text)
        # First display block (already delimited) preserved
        assert r"$$V = \int_{0}^{2\pi}" in result
        # Inline $\rho$ preserved
        assert "$\\rho$" in result
        # Bare equation lines wrapped
        lines = result.split("\n")
        rho_eq = [l for l in lines if "\\frac{R^3}{3}" in l][0].strip()
        assert rho_eq.startswith("$$")
        assert rho_eq.endswith("$$")


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and tricky inputs."""

    def test_empty_string(self):
        assert normalize_latex("") == ""

    def test_no_latex(self):
        assert normalize_latex("Hello, world!") == "Hello, world!"

    def test_code_block_not_touched(self):
        text = "```python\n\\frac{a}{b}\n```"
        assert normalize_latex(text) == text

    def test_subscript_dollar_stripped(self):
        result = strip_dollars_in_subscripts(r"_{$\infty$}")
        assert result == r"_{\infty}"

    def test_begin_end_environment_wrapped(self):
        text = r"\begin{align} x &= 1 \\ y &= 2 \end{align}"
        result = normalize_latex(text)
        assert "$$" in result

    def test_begin_end_already_in_dollars_not_double_wrapped(self):
        text = r"$$\begin{align} x &= 1 \end{align}$$"
        result = normalize_latex(text)
        assert result.count("$$") == 2  # just the original pair
