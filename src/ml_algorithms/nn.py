import numpy as np


class MultiLayerPerceptron():
    def __init__(self, inputs, targets, n_hidden, beta):
        self.n_data = len(inputs)
        self.inputs = np.concatenate((inputs, -np.ones((self.n_data, 1))), axis=1)
        self.targets = targets
        self.n_in = inputs.shape[1]  # Number of attributes
        self.n_out = targets.shape[1]  # Number of outputs
        self.n_hidden = n_hidden  # Number of hidden neurons
        self.beta = beta

        self.weights1 = np.random.randn(self.n_in + 1, n_hidden) * 0.1 - 0.05
        self.weights2 = np.random.randn(self.n_hidden + 1,
                                        self.n_out) * 0.1 - 0.05

    def train(self, eta, iterations):

        for n in range(iterations):

            self.outputs = self.fwd(self.inputs)
            error = 0.5 * sum((self.targets - self.outputs)**2)
            print("Iteration: " + str(n) + '\t' + "Error: " + str(error))

            deltao = (self.outputs - self.targets) * self.outputs * (1.0 - self.outputs)
            deltah = self.hidden * (1.0 - self.hidden) * \
                np.dot(-deltao, np.transpose(self.weights2))

            updatew1 = np.zeros((np.shape(self.weights1)))
            updatew2 = np.zeros((np.shape(self.weights2)))

            updatew1 = eta * np.dot(np.transpose(self.inputs), deltah[:, :-1])
            updatew2 = eta * np.dot(np.transpose(self.hidden), deltao)

            self.weights1 += updatew1
            self.weights2 -= updatew2

    def fwd(self, inputs):

        # If inputs are not the same as inputs stored in the nn object add column of ones
        if not np.array_equal(inputs, self.inputs):
            inputs = np.concatenate((inputs, -np.ones((len(inputs), 1))), axis=1)

        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden, -np.ones((inputs.shape[0], 1))), axis=1)
        outputs = np.dot(self.hidden, self.weights2)
        outputs = 1.0 / (1.0 + np.exp(-self.beta * outputs))

        return outputs

    def confmat(self, inputs, outputs):
        """
        UNFINISHED
        """

        nn_outputs = np.where(self.fwd(inputs) > 0.5, 1, 0)

        return nn_outputs
