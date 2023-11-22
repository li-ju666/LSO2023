import numpy as np
import time
from scipy.linalg import clarkson_woodruff_transform
import matplotlib.pyplot as plt


# load data
data = np.load("as2/song_preprocessed.npz")
A, b = data["A"], data["b"]


def get_value(A, b, x):
    return 1/b.shape[0] * np.linalg.norm(A @ x - b)**2


# 1. solve the linear system with original method
def original_solve(A, b):
    t = time.time()
    # solve the linear system
    x = np.linalg.solve(A.T@A, A.T@b)
    ori_time = time.time() - t
    return x, ori_time


# 2. solve the linear system with sketching method
def sketch_solve(A, b, sketch_size):
    t = time.time()
    # generate sketching matrix
    SAb = clarkson_woodruff_transform(
        np.hstack((A, b.reshape(-1, 1))),
        sketch_size)
    SA, Sb = SAb[:, :-1], SAb[:, -1]
    # solve the linear system
    x_sketch = np.linalg.solve(SA.T@SA, SA.T@Sb)
    sketch_time = time.time() - t
    return x_sketch, sketch_time


# 3. run experiments
x, ori_time = original_solve(A, b)

results = {}

sketch_sizes = [int(np.power(10, i)) for i in np.linspace(3, 5, 10)]
print(sketch_sizes)
for sketch_size in sketch_sizes:
    x_sketch, sketch_time = sketch_solve(A, b, sketch_size)
    results[sketch_size] = {
        "value": get_value(A, b, x_sketch),
        "time": sketch_time,
        "error": np.linalg.norm(x - x_sketch)
    }

# 4. plot the results
# 1). function value: compare original and sketching of different sizes
plt.figure()
plt.plot(sketch_sizes, [results[sketch_size]["value"]
                        for sketch_size in sketch_sizes],
         '-o', label=r"$f(\hat{x}_m)$")
plt.plot(sketch_sizes, [get_value(A, b, x)] * len(sketch_sizes),
         label=r"$f(x^\star)$")
plt.xlabel(r"Sketching $m$")
plt.ylabel(r"$f(x)$")
plt.title("Function value of the solutions")
plt.xscale("log")
plt.legend()
plt.savefig("as2/sketch_value.pdf", dpi=300)
plt.close()

# 2). time: compare original and sketching of different sizes
plt.figure()
plt.plot(sketch_sizes, [results[sketch_size]["time"]
                        for sketch_size in sketch_sizes],
         '-o', label="Sketching")
plt.plot(sketch_sizes, [ori_time] * len(sketch_sizes),
         label="Original")
plt.xlabel(r"Sketching $m$")
plt.ylabel("Time (s)")
plt.title("Time elapsed for solving the linear system")
plt.xscale("log")
plt.legend()
plt.savefig("as2/sketch_time.pdf", dpi=300)
plt.close()

# 3). error: compare original and sketching of different sizes
plt.figure()
plt.plot(sketch_sizes, [results[sketch_size]["error"]
                        for sketch_size in sketch_sizes],
         '-o')
plt.xlabel(r"Sketching $ m$")
plt.ylabel(r"$\| \hat{x}_m - x^\star \|$")
plt.title("Error of the sketching solutions")
plt.xscale("log")
plt.savefig("as2/sketch_error.pdf", dpi=300)
plt.close()
