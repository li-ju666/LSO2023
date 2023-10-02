import numpy as np
from scipy.linalg import cholesky, solve_triangular
import scipy.sparse
import time


# Create synthetic data
np.random.seed(0)
n = 5000
m = n
k = 5 * n
H = np.random.randn(m, n)
P = scipy.sparse.random(n, k, density=0.01).toarray()
y = H @ P + np.random.randn(m, k)
assert np.linalg.matrix_rank(P) == n

print(f"n={n}, m={m}, k={k}")

PT = P.T.copy()
A = P @ PT

t = time.time()
# factorize A with cholesky decomposition
L = cholesky(A, lower=True)
print(f"Computing L: {time.time() - t} seconds.")

# solve the linear systems
t = time.time()
Z = solve_triangular(
    L, P@y.T, lower=True, check_finite=False)
H_star = solve_triangular(
    L.T, Z, lower=False, check_finite=False)

print(f"Computing H_star: {time.time() - t} seconds.")

# use naive method to compute the residual
H_naive = np.linalg.solve(A, P@y.T)

# compute the residual
residual = np.linalg.norm(H_star - H_naive)
print(f"Residual: {residual}")
