import numpy as np
from scipy.linalg import cholesky, solve_triangular
import scipy.sparse
import time

t = time.time()
# Create synthetic data
np.random.seed(0)
n = 5000
m = n
k = 5 * n
H = np.random.randn(m, n)
P = scipy.sparse.random(n, k, density=0.01).toarray()
y = H @ P + np.random.randn(m, k)
assert np.linalg.matrix_rank(P) == n

print(f"Time elapsed for generating: {time.time() - t}s")

t = time.time()
PT = P.T.copy()
A = P @ PT

# factorize A with cholesky decomposition
L = cholesky(A, lower=True)

# solve the linear systems
Z = solve_triangular(
    L, P@y.T, lower=True, check_finite=False)
H_star = solve_triangular(
    L.T, Z, lower=False, check_finite=False).T

print(f"Time elapsed for solving: {time.time() - t}s")

# # use naive method to compute the residual
# H_naive = np.linalg.solve(A, P@y.T).T

# # compute the residual
# error = np.linalg.norm(H_star.flatten() - H_naive.flatten())
# print(f"Error: {error}")
