import pypamm
import numpy as np

X = np.random.rand(100, 3)
idx, Y = pypamm.select_grid_points(X, ngrid=10)
print(idx)
print(Y)