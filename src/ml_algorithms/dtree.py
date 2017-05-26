import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt


class BasicTree:

    def __init__(self):
        """
        Constructor of the basic decision tree

        :return:
        """

    @staticmethod
    def read_data(file_name):
        """
        Reads the data from a *.csv file and stores it in the data objects.
        Also, returns the read information.

        :param file_name: Path to the dataset
        :returns data: The data read from a file (values of the attributes)
        :returns classes: The classes that each of the examples takes
        :returns feature_names: Names of the attributes describing examples
        """

        fid = open(file_name, 'r')
        data = []
        classes = []
        feature_names = []
        for line_index, line in enumerate(fid.readlines()):
            line = line.strip()
            if line_index == 0:
                feature_names = line.split(',')[:-1]
            else:
                data.append(map(float, line.split(',')[:-1]))
                if float(line.split(',')[-1]) > 0:
                    classes.append(1)
                else:
                    classes.append(0)

        fid.close()

        return data, classes, feature_names

    @staticmethod
    @staticmethod
    def calc_entropy(p):
        """
        Calculates entropy from the provided probability

        .. math::
            -p \ \mathrm{log}_2(p)

        :param p: Probability of the feature value
        :return H: Returns the calculated entropy :math:`\\mathrm{H}(s)`
        """
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def calc_info(self, data, classes):
        """
        Calculates weighted entropy of a dataset:

        .. math::
            \\sum_{f \in F} \\frac{\\lvert S_f \\rvert }{\\lvert S \\rvert } \\mathrm{H}(s)


        :param data:
        :param classes:
        :return:
        """

        number_of_data = len(data)
        class_counter = Counter(classes)
        frequency = class_counter.values()
        info = reduce(lambda x, y: x + self.calc_entropy(
            float(y) / float(number_of_data)),  frequency, 0)

        return info

    def calc_info_gain(self, data, classes, feature):
        """
        Calculates the information gain for all of the thresholds

        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature: The feature for which the information gain is calculated
        """
        n_data = len(data)

        # If all feature values are equal, no point in splitting data
        feature_column = []
        for datapoint in data:
            feature_column.append(datapoint[feature])
        if len(set(feature_column)) == 1:
            return [-1], [-1]

        # Extract unique values of feature

        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        # Sort the values in ascending order

        values.sort()

        # Create thresholds in the middle of the data

        thresholds = np.zeros(len(values) - 1)
        for value_index in range(len(values) - 1):
            thresholds[value_index] = values[value_index] + \
                (values[value_index + 1] - values[value_index]) / 2

        # Calculate total entropy

        total_entropy = self.calc_info(data, classes)

        # Find the lower and upper dataset

        gain = np.zeros(len(thresholds))
        for threshold_index, threshold in enumerate(thresholds):
            entropy = 0

            # Split the data for each threshold

            lower_data, lower_classes, upper_data, upper_classes = self.split_data(
                data, classes, threshold, feature)

            entropy += float(len(lower_data)) / float(n_data) * \
                self.calc_info(lower_data, lower_classes)
            entropy += float(len(upper_data)) / float(n_data) * \
                self.calc_info(upper_data, upper_classes)

            # Calculate total gain

            gain[threshold_index] = total_entropy - entropy

        return gain, thresholds

    @staticmethod
    def split_data(data, classes, threshold, feature_index):
        """
        Splits the data object according to the provided threshold of a selected attribute

        :param data: Dataset containing examples
        :param classes: Classes of the examples in a dataset
        :param threshold: Threshold according to which the split is made
        :param feature_index: Feature according to which the split is made
        :return: Returns lower and upper data and corresponding classes
        """

        lower_data = []
        lower_classes = []
        upper_data = []
        upper_classes = []

        for data_index, datapoint in enumerate(data):
            if datapoint[feature_index] <= threshold:
                lower_data.append(datapoint)
                lower_classes.append(classes[data_index])
            elif datapoint[feature_index] > threshold:
                upper_data.append(datapoint)
                upper_classes.append(classes[data_index])

        return lower_data, lower_classes, upper_data, upper_classes

    def make_tree(self, data, classes, feature_names):
        """
        Creates a tree dictionary from the dataset

        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature_names: Names of the features which are observed
        :return tree: A graph describing the constructed tree
        """

        # Find the number of features and data

        number_of_features = len(feature_names)
        number_of_data = len(data)

        # Calculate the frequency of each class value in the current data set

        class_counter = Counter(classes)
        frequency = class_counter.values()

        # If the dataset contains points of only a single class, we reached the
        # leaf (return the class)

        if classes.count(classes[0]) == number_of_data:
            return classes[0]

        # If there is no more data, then return the most frequent class at the previous node

        elif number_of_data == 0 or number_of_features == 0:
            return classes[np.argmax(frequency)]

        else:
            threshold = 0
            gain = -1
            best_feature_index = -1
            for feature_index in range(number_of_features):
                # Find the maximum gain and threshold for splitting the data set
                gain_list, thresholds_list = self.calc_info_gain(data, classes, feature_index)
                th_index = np.argmax(gain_list)
                if gain_list[th_index] > gain:
                    gain = gain_list[th_index]
                    threshold = thresholds_list[th_index]
                    best_feature_index = feature_index

            tree = {feature_names[best_feature_index]: {}}

            lower_data, lower_classes, upper_data, upper_classes = self.split_data(
                data, classes, threshold, best_feature_index)

            subtree_lower = self.make_tree(lower_data, lower_classes, feature_names)
            subtree_upper = self.make_tree(upper_data, upper_classes, feature_names)

            tree[feature_names[best_feature_index]]['>'] = [threshold, subtree_upper]
            tree[feature_names[best_feature_index]]['<='] = [threshold, subtree_lower]

            return tree

    def print_tree(self, tree, ind=""):
        """
        Prints the decision tree

        :param tree: The tree dictionary
        :param ind: Optional parameter used to make indentation in the printout
        :return:
        """
        if type(tree) == dict:
            for key in tree.keys():
                if key == '<=':
                    print ind, "|", key, str(tree[key][0])
                    self.print_tree(tree[key][1], ind + "\t")
                elif key == '>':
                    print ind, "|", key, str(tree[key][0])
                    self.print_tree(tree[key][1], ind + "\t")
                else:
                    print ind, "|", key
                    self.print_tree(tree[key], ind + "\t")
        else:
            print ind, "->", tree

    def plot_results(self, training_data, training_classes, test_data, test_classes, test_results):
        """
        Plots the classification results

        :param training_data:
        :param training_classes:
        :param test_data:
        :param test_classes:
        :param test_results:
        :return:
        """
        colors = []
        x = [point[0] for point in training_data]
        y = [point[1] for point in training_data]
        for a_class in training_classes:
            if a_class == 1:
                colors.append('#59d286')
            else:
                colors.append('#fe6447')

        plt.scatter(x, y, s=20, c=colors)
        plt.xlabel('Active power (MW)')
        plt.ylabel('Reactive power (Mvar)')

        colors = []
        x = [point[0] for point in test_data]
        y = [point[1] for point in test_data]
        for test_class, test_result in zip(test_classes, test_results):
            if test_class == test_result:
                colors.append('#59d286')
            else:
                colors.append('#fe6447')
        plt.scatter(x, y, marker='x', s=40, c=colors)
        plt.draw()
        plt.show()

    def classify(self, tree, datapoint, feature_names):
        """
        Classifies the data point using the decision tree

        :param tree: Dictionary of a decision tree
        :param datapoint: Data point to be classified
        :param feature_names: Feature names of the data
        :return data_class: Class of the data point
        """

        if type(tree) == dict:
            for key in tree.keys():
                for feature_index, feature_name in enumerate(feature_names):
                    if feature_name == key:
                        if datapoint[feature_index] <= tree[key]['<='][0]:
                            return self.classify(tree[key]['<='][1], datapoint, feature_names)
                        elif datapoint[feature_index] > tree[key]['>'][0]:
                            return self.classify(tree[key]['>'][1], datapoint, feature_names)
        else:
            return tree

    def classify_set(self, tree, data, feature_names):
        """
        Classifies a data set using the provided tree dictionary

        :param tree:
        :param data:
        :param feature_names:
        :return:
        """

        classes = []
        for datapoint in data:
            classes.append(self.classify(tree, datapoint, feature_names))

        return classes

    def tree_test(self, tree, test_data, test_classes, feature_names):
        """
        Test the decision tree using the test data and print the statistics

        :param tree:
        :param test_data:
        :param test_classes:
        :param feature_names:
        :return:
        """

        result_classes = []
        result_classes = self.classify_set(tree, test_data, feature_names)

        true_results = [x == y for (x, y) in zip(result_classes, test_classes)]

        return float(sum(true_results)) / float(len(test_classes)), result_classes
