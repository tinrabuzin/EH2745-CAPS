import numpy as np


class BasicTree:

    def __init__(self):
        """
        Constructor
        :return:
        """
        self.data = [['Urgent', 'Yes', 'Yes'], ['Urgent', 'No', 'Yes'], ['Near', 'Yes', 'Yes'], ['None', 'Yes', 'No'], ['None', 'No', 'Yes'], ['None', 'Yes', 'No'], ['Near', 'No', 'No'], ['Near', 'No', 'Yes'], ['Near', 'Yes', 'Yes'], ['Urgent', 'No', 'No']]
        self.classes = ['Party', 'Study', 'Party', 'Party', 'Pub', 'Party', 'Study', 'TV', 'Party', 'Study']
        self.features = ['Deadline', 'Party', 'Lazy']

    @staticmethod
    def calc_entropy(p):
        """
        Calculates the entropy of a feature using its probability
        :param p: Probability of the feature
        :return: Returns the calculated entropy
        """
        if p != 0:
            return -p*np.log2(p)
        else:
            return 0

    def calc_info_gain(self, data, classes, feature):
        """
        Calculates the information gain
        :param self:
        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature: Feature index in the data
        :return gain: Calculated gain for selecting a certain feature
        """
        gain = 0
        n_data = len(data)

        # Values that the selected feature can take (read from the dataset)
        values = []
        for data_point in data:
            if data_point[feature] not in values:
                values.append(data_point[feature])

        feature_counts = np.zeros(len(values))
        entropy = np.zeros(len(values))

        # Iterate through the dataset and find classes that appear with the selected feature value

        for value_index, value in enumerate(values):
            new_classes = []
            for data_index, data_point in enumerate(data):
                if data_point[feature] == value:
                    feature_counts[value_index] += 1
                    new_classes.append(classes[data_index])

            # Find class values that a class can take corresponding to the current feature value

            class_values = list(set(new_classes))

            # Find a number of occurrences of a certain class value

            class_counts = np.zeros(len(class_values))

            for class_index, classValue in enumerate(class_values):
                class_counts[class_index] = new_classes.count(classValue)

            # Calculate a gain

            for class_index in range(len(class_values)):
                entropy[value_index] += self.calc_entropy(float(class_counts[class_index])/sum(class_counts))
            gain += float(feature_counts[value_index])/n_data * entropy[value_index]

        return gain

    def make_tree(self, data, classes, feature_names):
        """
        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature_names: Names of the features which are observed
        :return tree: A graph describing the constructed tree
        """

        number_of_features = len(data[0])
        number_of_data = len(data)

        # Finding possible classes

        new_classes = []
        for aclass in classes:
            if new_classes.count(aclass) == 0:
                new_classes.append(aclass)

        # Calculate total entropy

        total_entropy = 0

        for aclass in new_classes:
            total_entropy += self.calc_entropy(float(classes.count(aclass))/float(number_of_data))

        # Check if we came to an end

        if number_of_data == 0 or number_of_features == 0:
            return 0
        elif classes.count(classes[0]) == number_of_data:
            return classes[0]
        else:

            # Find the feature with the greatest information gain

            gain = np.zeros(number_of_features)
            for feature_index in range(number_of_features):
                g = self.calc_info_gain(data, classes, feature_index)
                gain[feature_index] = total_entropy - g

            best_feature_index = np.argmax(gain)
            tree = {feature_names[best_feature_index]}
            values = []

            # Extract the feature values that the datapoints take

            for datapoint in data:
                if values.count(datapoint[best_feature_index]) == 0:
                    values.append(datapoint[best_feature_index])

            for value in values:
                new_data = []
                new_classes = []
                index = 0

                # Extract the data which has the current feature value

                for data_index, datapoint in enumerate(data):
                    if datapoint[best_feature_index] == value:
                        datapoint.pop(best_feature_index)
                        new_feature_names = list(feature_names)
                        new_feature_names.pop(best_feature_index)
                        new_data.append(datapoint)
                        new_classes.append(classes[data_index])

                # Create a new subtree out of remaining data

                subtree = self.make_tree(new_data, new_classes, new_feature_names)

                # When coming back up merge the trees

                tree[feature_names[best_feature_index]][value] = subtree

                return tree
