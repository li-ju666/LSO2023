import numpy as np

# load data
data = np.load("as2/song_preprocessed.npz")
A, b = data["A"], data["b"]

# solve the linear system
x = np.linalg.solve(A.T@A, A.T@b)

# print residual
# print(np.linalg.norm(A@x - b)**2/b.shape[0])

print(A@x - b)
