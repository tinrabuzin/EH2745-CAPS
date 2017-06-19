import ml_algorithms
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [0], [0], [1]])

data_obj = ml_algorithms.Data('/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv')
inputs, classes, feature_names = data_obj.read_data()
targets = data_obj.read_tcc('/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset_tcc.csv')

nn = ml_algorithms.MultiLayerPerceptron(np.array(inputs), np.array(targets), 5, 1)
nn.train(0.25, 1000)

# print "FwdResults"
#pn.confmat([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
