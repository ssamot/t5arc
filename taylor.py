import sympy as sp


def taylor_expansion_2d(f, a, b, x, y, order=2):
    """
    Compute the Taylor expansion of a function f(x, y) around the point (a, b).

    Parameters:
    - f: sympy function of two variables x and y
    - a: x-coordinate of the expansion point
    - b: y-coordinate of the expansion point
    - x: sympy symbol for the x variable
    - y: sympy symbol for the y variable
    - order: order of the Taylor expansion (default is 2)

    Returns:
    - Taylor expansion as a sympy expression
    """
    expansion = f.subs({x: a, y: b})
    print(expansion, x, a, y, b)

    for i in range(1, order + 1):
        for j in range(i + 1):
            term = sp.diff(f, x, j, y, i - j) / (sp.factorial(j) * sp.factorial(i - j)) * (x - a) ** j * (y - b) ** (
                        i - j)
            expansion += term.subs({x: a, y: b})
            print(expansion)

    return expansion


# Example usage
x, y = sp.symbols('x y')
f_expr = x ** 2 + y ** 2 + x * y  # Example function
a, b = 1, 1  # Expansion point

taylor_series = taylor_expansion_2d(f_expr, a, b, x, y)
print(taylor_series)
