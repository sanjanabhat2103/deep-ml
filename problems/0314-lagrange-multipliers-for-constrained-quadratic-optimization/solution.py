import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.

    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float).reshape(2)
    a = np.asarray(a, dtype=float).reshape(2)

    # Construct the KKT system
    KKT = np.block([
        [Q, a.reshape(-1, 1)],
        [a.reshape(1, -1), np.zeros((1, 1))]
    ])

    rhs = np.concatenate((-c, [b]))

    # Solve for x and lambda
    sol = np.linalg.solve(KKT, rhs)
    x = sol[:2]
    lam = sol[2]

    # Compute objective value
    obj = 0.5 * x @ Q @ x + c @ x

    return {
        "x": np.round(x, 4).tolist(),
        "lambda": -round(float(lam), 4),
        "objective": round(float(obj), 4),
    }
