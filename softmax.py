
import numpy as np

a = np.array([1,5,7])

softmax_a = np.exp(a) / np.sum(np.exp(a))
print(softmax_a)

