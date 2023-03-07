import numpy as np
import random

random.seed(1)

orig = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        orig[i, j] = random.randint(0, 9)

print(orig)
flattend = orig.flatten()
print(flattend)
print(flattend.reshape(4, 4))