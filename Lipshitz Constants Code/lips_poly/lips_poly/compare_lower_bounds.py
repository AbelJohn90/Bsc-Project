import numpy as np
import math

n = 1
d = 10
h = 100

W = np.random.randn(100, 10)
v = np.random.randn(1, 100)

vW = v @ W

norm2 = np.linalg.norm(vW, ord=2)
estimated_l1 = math.sqrt(d) * norm2
norm1 = np.linalg.norm(vW, ord=1)
print(norm2, estimated_l1, norm1)

