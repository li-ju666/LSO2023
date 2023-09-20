import numpy as np
import scipy.sparse as sp
import matplotlib.pylab as plt


# The problem we are solving is
# $\arg\min_{x} 1/2 x^\top L x - b^\top x$
# Note that $L$ is symmetric and positive semi-definite.


def get_problem():
    np.random.seed(42)
    n_nodes = 10000
    n_deg = 20
    # compute matrix density
    density = n_deg/n_nodes
    # Right hand side vector (uniform)
    b = np.random.rand(n_nodes)
    # Construct matrix
    # conductance uniform [0,1]
    L = -np.abs(sp.rand(n_nodes+1, n_nodes+1, density=density,
                        format="coo", random_state=42))
    # make L symmetric
    L = sp.triu(L) + sp.triu(L, 1).T
    # Finalize diagonal
    L = L - sp.diags(L@np.ones(n_nodes+1), format="coo")
    # remove last row and column
    L = L[:-1, :-1]
    print(L.shape, b.shape)
    return L, b


def get_residual(L, x, b):
    return L@x - b


def conjugate_gradient_descent(L, b, x0, eps, tmax):
    # Initialize
    x = x0
    r = get_residual(L, x, b)
    p = -r

    residuals = [np.sqrt(r.T@r)]
    # Iterate
    t = 0
    while t < tmax:
        alpha = (r.T@r)/(p.T@L@p)
        x = x + alpha*p
        r_new = r + alpha*L@p
        beta = (r_new.T@r_new)/(r.T@r)
        p = -r_new + beta*p
        r = r_new

        residual = np.sqrt(r.T@r)
        residuals.append(residual)
        if residual < eps:
            break
        t += 1
    print(f"Residual (||Lx - b||): {residual}")
    return np.array(residuals)


results = conjugate_gradient_descent(
    *get_problem(), np.zeros(10000), 1e-6, 10000)

# plot results
plt.plot(np.log10(results), label="CGD")

plt.xlabel("Num of iterations")
plt.ylabel(r"$\log_{10}(\| r \|)$")
plt.title("Convergence of CGD")
# plt.ylim(-6, 1)

plt.legend()
plt.savefig("probcg.pdf", dpi=300)
