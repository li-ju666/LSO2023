import numpy as np
import matplotlib.pylab as plt


# script to compare different optimization methods
# for finding the minimal for $\frac{1}{2} x^T A x$

n = 100
m = 0.
L = 1
epsilon = 1e-6
tmax = 1000


# generate matrix A with eigenvalues in [m, L]
def get_A_matrix(n=100, m=0.01, L=1):
    n = 100
    np.random.seed(42)
    A = np.random.randn(n, n)
    (Q, R) = np.linalg.qr(A)

    D = np.random.rand(n)
    D = 10.**D
    Dmin = D.min()
    Dmax = D.max()
    D = (D-Dmin)/(Dmax-Dmin)
    D = m + D*(L-m)
    A = Q.T@np.diag(D)@Q
    return A


def get_grad(A, x):
    grad = A@x.T
    return grad


def get_value(A, x):
    value = 1/2*x@A@x.T
    return value


def steepest_descent_fixed_stepsize(A, x0, eta, epsilon, tmax):
    x = x0
    t = 0

    fxs = [get_value(A, x)]
    while t < tmax:
        grad = get_grad(A, x)
        x = x - eta*grad
        fx = get_value(A, x)
        if fx < epsilon:
            break
        # print(f"t={t} with f(x)={fx}")
        t += 1
        fxs.append(fx)
    return np.array(fxs)


def steepest_descent_line_search_stepsize(A, x0, epsilon, tmax):
    x = x0
    t = 0
    fxs = [get_value(A, x)]
    while t < tmax:
        grad = get_grad(A, x)
        eta = np.linalg.norm(grad)**2/(grad.T@A@grad)
        x = x - eta*grad
        fx = get_value(A, x)
        if fx < epsilon:
            break
        # print(f"t={t} with f(x)={fx}")
        t += 1
        fxs.append(fx)
    return np.array(fxs)


def heavy_ball(A, x0, eta, beta, epsilon, tmax):
    x_prev = x0
    x = x0
    t = 0
    fxs = [get_value(A, x)]
    while t < tmax:
        grad = get_grad(A, x)
        x_next = x - eta*grad + beta*(x - x_prev)
        x_prev = x
        x = x_next

        fx = get_value(A, x)
        if fx < epsilon:
            break
        # print(f"t={t} with f(x)={fx}")
        t += 1
        fxs.append(fx)
    return np.array(fxs)


def nesterov(A, x0, eta, beta, epsilon, tmax):
    x = x0
    last_update = 0
    t = 0
    fxs = [get_value(A, x)]
    while t < tmax:
        tmpx = x + beta*last_update
        grad = get_grad(A, tmpx)
        x_next = x - eta*grad + beta*last_update
        last_update = x_next - x
        x = x_next

        fx = get_value(A, x)
        if fx < epsilon:
            break
        # print(f"t={t} with f(x)={fx}")
        t += 1
        fxs.append(fx)
    return np.array(fxs)


# using the same matrix A for all runs but different x0
def one_run(seed=42):
    A = get_A_matrix()

    np.random.seed(seed)
    x0 = np.random.randn(n)

    results = {
        r"SD: $\eta=\frac{2}{L+m}$":
            steepest_descent_fixed_stepsize(A, x0, 2/(L+m), epsilon, tmax),
        r"SD: $\eta=\frac{1}{L}$":
            steepest_descent_fixed_stepsize(A, x0, 1/L, epsilon, tmax),
        "SD: Line Search":
            steepest_descent_line_search_stepsize(A, x0, epsilon, tmax),
        "Heavy Ball Momen.":
            heavy_ball(A, x0,
                    #    4/(L+m+2*np.sqrt(m*L)),
                    #    (np.sqrt(L)-np.sqrt(m))/(np.sqrt(L)+np.sqrt(m)),
                       1/L, 0.9,
                       epsilon, tmax),
        "Nesterov Momen.":
            nesterov(A, x0, 1/L,
                    #  (np.sqrt(L)-np.sqrt(m))/(np.sqrt(L)+np.sqrt(m)),
                     0.9,
                     epsilon, tmax)
    }
    return results


# generate the result table with 10 different runs
num_iterations = {}
random_seeds = np.random.randint(0, 1000, 10)
for seed in random_seeds:
    results = one_run(seed)
    for key, value in results.items():
        if key not in num_iterations:
            num_iterations[key] = []
        num_iterations[key].append(len(value))

# formulate the table and print it
print("Method & Mean & Std. Dev. \\\\")
for key, value in num_iterations.items():
    print(f"{key} & {np.mean(value):.2f} & {np.std(value):.2f} \\\\")
print("")
print("")  # empty line

results = one_run(1)
# plot results with log10 scale
for key, value in results.items():
    plt.plot(np.log10(value), label=key)


plt.xlabel("Num of iterations")
plt.ylabel(r"$\log_{10}(f(x) - f(x^\star)$")
plt.title("Convergence of 1st-order Methods")
plt.ylim(-6, 1)

plt.legend()
plt.savefig("prob42.pdf", dpi=300)
