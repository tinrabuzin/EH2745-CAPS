import ml_algorithms
import numpy as np


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])
pn = ml_algorithms.PerceptronNetwork(inputs, targets)
pn.train(0.25, 6)
