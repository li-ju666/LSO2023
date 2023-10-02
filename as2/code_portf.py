import numpy as np
import time


# generate some random data
n = 15000  # number of assets
k = 30  # number of factors
np.random.seed(0)
F = np.random.randn(n, k)
F = np.matrix(F)
d = 0.1 + np.random.rand(n)
d = np.matrix(d).T
Q = np.random.randn(k)
Q = np.matrix(Q).T
Q = Q * Q.T + np.eye(k)
Sigma = np.diag(d.A1) + F*Q*F.T
mu = np.random.rand(n)
mu = np.matrix(mu).T


# the slow way, solve full KKT
t = time.time()
kkt_matrix = np.vstack((np.hstack((Sigma, np.ones((n, 1)))),
                        np.hstack((np.ones(n), [0.]))))

wnu = np.linalg.solve(kkt_matrix, np.vstack((mu, [1.])))
print(f"Elapsed time for naive method is {(time.time() - t)} seconds.")
wslow = wnu[:n]

# fast method: solve the linear system with block matrix
t = time.time()

dinv = np.asarray(1./d)
Qinv = np.linalg.inv(Q)

# 1. formulation step:
dinvF = np.multiply(dinv, F)
dinvmu = np.multiply(dinv, mu)

# Compuet S and $\tilde{d}$
s11 = np.array([[dinv.sum()]])
s12 = np.ones((1, n))@dinvF
s21 = s12.T
s22 = F.T @ dinvF + Qinv


S = -np.vstack((np.hstack((s11, s12)),
                np.hstack((s21, s22))))

# Compute $\tilde{b}$
tilde_b = np.concatenate(
    (
        np.array([[1 - dinvmu.sum()]]),
        -F.T@dinvmu,
    )
)

# 2. solving step:

# Solve the linear system
# 1). solve Sx_2 = \tilde{b}
x_2 = np.linalg.solve(S, tilde_b)

# 2). solve x1 (i.e. (w, y) ) using x2
b1 = np.concatenate((mu, np.zeros((k, 1))))

A12 = np.vstack(
    (np.hstack((np.ones((n, 1)), F)),
     np.hstack((np.zeros((k, 1)), -np.eye(k))))
)

right_hand_side = b1 - A12@x_2

wfast = np.multiply(dinv, right_hand_side[:n])

print(f"Time for solving is {(time.time() - t)} seconds.")
rel_err = np.sqrt(np.sum((wfast-wslow).A1**2)/np.sum(wslow.A1**2))
print(f"Error: {rel_err}")
