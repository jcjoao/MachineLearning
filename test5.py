import numpy as np

y = np.load("NADAMAU-1260-500.npy")

print(y.shape)
print("How many 0s: ", np.count_nonzero(y == 0))
print("How many 1s: ", np.count_nonzero(y == 1))