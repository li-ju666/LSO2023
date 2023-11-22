# code for l1 regularized linear regression
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass


# read text file
def read_data(filename):
    data = np.loadtxt(filename)
    return data


def standarize(A):
    # standarize each column of A
    min_val = A.min(axis=0)
    max_val = A.max(axis=0)

    standarized = []
    for i in range(A.shape[1]):
        # skip if all values are the same
        if min_val[i] == max_val[i]:
            continue
        else:
            standarized.append((A[:, i] - min_val[i])/(max_val[i] - min_val[i]))
    return np.array(standarized).T


@dataclass
class L1LRSolver:
    A: np.ndarray
    b: np.array
    lam: float
    step_size: float
    max_step: int
    x: np.array = None

    def get_value(self):
        N = self.A.shape[0]
        residual = self.A@self.x - self.b
        t1 = 0.5 * (residual.T @ residual)/N
        t2 = self.lam * np.abs(self.x).sum()

        return t1 + t2

    def get_gradient(self):
        N = self.A.shape[0]
        grad = (self.A.T @ (self.A @ self.x - self.b))/N
        return grad


class SubgradientDescent(L1LRSolver):
    def get_subgradient(self):
        # compute gradient of T1
        t1_grad = self.get_gradient()

        # compute subgradient of T2
        t2_grad = self.lam * np.sign(self.x)
        # a random subgradient is sampled if x = 0 from [-1, 1]
        num_zeros = np.sum(self.x < 1e-7)
        t2_grad[self.x < 1e-7] = np.random.uniform(-1, 1, num_zeros)

        subgrad = t1_grad + t2_grad
        return subgrad

    def step(self):
        # compute subgradient
        subgrad = self.get_subgradient()
        # update x
        self.x -= self.step_size * subgrad

        # compute value
        value = self.get_value()

        return value

    def solve(self):
        values = []
        for i in range(self.max_step):
            value = self.step()
            values.append(value)
        return np.array(values)


class ProximalGradientDescent(L1LRSolver):
    def prox_map(self):
        proxed = np.sign(self.x) * np.maximum(np.abs(self.x) - self.lam * self.step_size, 0.)
        return proxed

    def step(self):
        # compute gradient
        grad = self.get_gradient()
        # update x
        self.x -= self.step_size * grad
        # proximal map
        self.x = self.prox_map()

        # compute value
        value = self.get_value()

        return value

    def solve(self):
        values = []
        for i in range(self.max_step):
            value = self.step()
            values.append(value)
        return np.array(values)


def main():
    data = read_data('./data/data.txt')

    A = data[:, :-2]
    A = standarize(A)
    b = data[:, -2]

    lam = 0.1
    max_step = 3000
    step_size = 1e-2

    # start from zero
    results = {}

    x = np.zeros(A.shape[1])
    step_size = 1e-2
    prox_solver = ProximalGradientDescent(A, b, lam, step_size, max_step, x)
    results[f'Proximal: $\eta={step_size}$'] = prox_solver.solve()

    x = np.zeros(A.shape[1])
    step_size = 1e-2
    sub_solver = SubgradientDescent(A, b, lam, step_size, max_step, x)
    results[f'Subgradient: $\eta={step_size}$'] = sub_solver.solve()

    x = np.zeros(A.shape[1])
    step_size = 2e-3
    sub_solver = SubgradientDescent(A, b, lam, step_size, max_step, x)
    results[f'Subgradient: $\eta={step_size}$'] = sub_solver.solve()

    # plot
    for key in results:
        plt.plot(results[key], label=key)
    plt.xlabel('Num. step')
    plt.ylabel(r'$f(x)$')
    plt.title("Subgradient vs Proximal\n$\ell_1$ regularized linear regression")
    plt.legend()

    plt.savefig('./l1lr.pdf', dpi=300)


if __name__ == '__main__':
    main()
