import numpy as np
from matplotlib import pyplot as plt


def project_onto_unit_ball(x):
    """Project point x onto unit ball."""
    l2norm = np.linalg.norm(x)
    x = x / l2norm if l2norm > 1 else x
    return x


def project_onto_unit_simplex(x):
    """Project point x onto simplex.
    Reference:
    https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    """
    # get the dimension of x
    dim = len(x)

    # sort x in descending order
    mu = np.sort(x)[::-1]

    # calculate the cumulative sum of mu
    cumm_sum_mu = np.cumsum(mu)

    # find rho
    indices = np.arange(dim) + 1
    conditions = (mu * indices - (cumm_sum_mu - 1) > 0)
    rho = indices[conditions][-1]

    # calculate theta
    theta = (cumm_sum_mu-1)[rho-1] / float(rho)

    projected_x = np.maximum(x - theta, 0)

    return projected_x


def project_onto_unit_box(x):
    """Project point x onto unit box"""
    x = np.maximum(x, 0)
    x = np.minimum(x, 1)
    return x


def projection_gradient_descent_step(theta, grad, learning_rate, project_fn):
    theta -= learning_rate * grad
    theta = project_fn(theta)
    return theta


def main():
    # solve $\min c^\top x$ with different constraints:
    # a). unit ball
    # b). unit simplex
    # c). unit box
    configs = {
        "Unit Ball": project_onto_unit_ball,
        "Unit Simplex": project_onto_unit_simplex,
        "Unit Box": project_onto_unit_box,
    }

    cs = [[0.3, 0.2], [-0.3, -0.2], 
          [0, 0.2], [0, -0.2], 
          [0.2, 0.2], [-0.2, -0.2]]
    # cs = cs[:1]
    cs = list(map(np.array, cs))

    max_steps = 1000
    for c in cs:
        paths = {}
        for constraint, proj_fn in configs.items():
            theta = np.array([4., 2.])
            learning_rate = 0.05
            grad = c
            paths[constraint] = []
            for _ in range(max_steps):
                paths[constraint].append(theta)
                theta = projection_gradient_descent_step(
                    theta, grad, learning_rate, proj_fn)
            print(f"Constraint: {constraint}", theta)

        # plot in 2D with ratio 4:4
        plot_lim = 2.5
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_xlim(-plot_lim, plot_lim)
        ax.set_ylim(-plot_lim, plot_lim)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        grad = f"({grad[0]},{grad[1]})"
        ax.set_title(f"Projected Gradient Descent: c={grad}")

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for idx, (constraint, path) in enumerate(paths.items()):
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], linewidth=0.8,
                    label=constraint, color=colors[idx],)

        # plot unit ball constraint
        ax.add_artist(
            plt.Circle((0, 0), 1, linestyle="--",
                       fill=False, color=colors[0], alpha=0.5))
        # plot unit simplex constraint
        ax.plot([0, 1], [1, 0], "--", color=colors[1], alpha=0.5)
        # plot unit box constraint
        ax.plot([0, 1], [0, 0], "--", color=colors[2], alpha=0.5)
        ax.plot([0, 0], [0, 1], "--", color=colors[2], alpha=0.5)
        ax.plot([1, 1], [0, 1], "--", color=colors[2], alpha=0.5)
        ax.plot([0, 1], [1, 1], "--", color=colors[2], alpha=0.5)

        ax.legend()
        plt.savefig(f"as3/projected_gd_{grad}.pdf", dpi=300)


if __name__ == "__main__":
    main()
