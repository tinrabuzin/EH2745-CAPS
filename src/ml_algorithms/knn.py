import numpy as np
import operator


class KNN:

    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(x, y):
        """
        Calculates the distance between two data points

        ...math::
            d = \sqrt{\sum(x_i-y_i)^2}

        :param x: Attributes of the data point x
        :param y: Attributes of the data point y
        :return d: Euclidean distance between the two points
        """
        d = np.sqrt(np.sum(np.square(x - y)))
        return d

    def neighbours_distance(self, data, point):
        """
        Calculates the distances between a data point and all of the points from a training set.
        It also sorts the distances beginning with the smallest one.

        :param data: Training set data attributes
        :param point: A point to which the distances are calulated
        :return distances: Distances from all of the training set points and a point
        """

        # Create a list for storing distances
        distances = []

        for n in range(len(data)):
            # First we compute the euclidean distance from a point to all of the training set
            distances.append([self.euclidean_distance(data[n], point), n])

        # Sort the list from the lowest to highest
        distances.sort(key=lambda x: x[0])

        return distances

    def classify(self, k, distances, data, classes):
        """
        Classifies the object (o_i) depending on k-value

        .. math::
            -> most NN classies gives the class (k/2+1)

        :param distances (from o_i):
        :param data (data_train):
        :param k:
        :return: class prediction for k
        """

        # Classify data depending on k-NN
        class_votes = {}
        for i in range(k):
            vote = classes[distances[1]]
            if vote in class_votes:
                class_votes[vote] += 1
            else:
                class_votes[vote] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)

        # Return most common target
        return sorted_votes[0][0]

    def calc_correct(data, Y_pred):
        """
        Calculates the correct predicitons in percentage

        .. math::
            ->(correct)/tot

        :param data:
        :param Y_pred:
        :return: correct amount in percenatge
        """
        correct = 0

        for i in range(len(data)):
            if Y_pred[i] == data[i][2]:
                correct = correct + 1

        correct_perc = correct / i

        return correct_perc
