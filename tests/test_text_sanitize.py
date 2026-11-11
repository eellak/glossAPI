from glossapi.text_sanitize import accept_latex, sanitize_latex, tail_run


def test_accept_latex_balanced():
    assert accept_latex(r"\frac{a}{b}") == 1.0
    assert accept_latex(r"\int_{0}^{1} x^2 dx") == 1.0
    assert accept_latex(r"\alpha + \beta = \gamma") == 1.0


def test_accept_latex_piecewise_and_escaped_braces():
    # Piecewise formula with \left\{ and \right.
    piecewise = r"f(x) = \left\{ \begin{matrix} 1 & x > 0 \\ 0 & \text{otherwise} \end{matrix} \right."
    assert accept_latex(piecewise) == 1.0

    # Set builder notation with escaped braces
    set_notation = r"A = \{ x \in \mathbb{R} \mid x > 0 \}"
    assert accept_latex(set_notation) == 1.0

    # Escaped right brace with dummy left delimiter
    delim_right = r"\left. \frac{df}{dx} \right\vert_{x=0} = 1"
    assert accept_latex(delim_right) == 1.0


def test_accept_latex_unbalanced_braces():
    assert accept_latex(r"\frac{a}{b") == 0.0
    assert accept_latex(r"\frac{a}{b}}") == 0.0
    assert accept_latex(r"}{") == 0.0


def test_accept_latex_dangerous_tokens():
    assert accept_latex(r"\includegraphics{image.png}") == 0.0
    assert accept_latex(r"\write18{rm -rf /}") == 0.0


def test_accept_latex_size_guard():
    assert accept_latex("x" * 6001) == 0.0
    assert accept_latex("x" * 100) == 1.0
