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

    def read_data(self, file_name):
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
                    classes.append('Secure')
                else:
                    classes.append('NotSecure')
        return data, classes, feature_names


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

    def calc_info(self, data, classes):
        """
        Calculates amount of information
        :param data:
        :param classes:
        :return:
        """

        number_of_data = len(data)

        info = 0

        class_values = []
        for value in classes:
            if class_values.count(value) == 0:
                class_values.append(value)

        for class_value in classes:
            info += self.calc_entropy(float(classes.count(class_value))/float(number_of_data))

        return info

    def calc_info_gain_cont(self, data, classes, feature):
        """
        Calculates the information gain for all of the thresholds

        """
        n_data = len(data)

        # Extract unique values of feature

        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        # Sort the values in ascending order

        values.sort()

        # Create thresholds in the middle of the data

        thresholds = np.zeros(len(values)-1)
        for value_index in range(len(values)-1):
            thresholds[value_index] = values[value_index] + (values[value_index+1] - values[value_index])/2

        # Do the data test
        entropy = 0
        lower_data = []
        lower_classes = []
        upper_data = []
        upper_classes = []

        # Find the lower and upper dataset
        gain = np.zeros(len(thresholds))
        for threshold_index, threshold in enumerate(thresholds):
            for data_index, datapoint in enumerate(data):
                if datapoint[feature] <= threshold:
                    lower_data.append(datapoint)
                    lower_classes.append(classes[data_index])
                elif datapoint[feature] > threshold:
                    upper_data.append(datapoint)
                    upper_classes.append(classes[data_index])

            # Calculate total entropy

            total_entropy = self.calc_info(data, classes)
            entropy += float(len(lower_data))/float(n_data)*self.calc_info(lower_data, lower_classes)
            entropy += float(len(upper_data))/float(n_data)*self.calc_info(upper_data, upper_classes)

            # Calculate total gain

            gain[threshold_index] = total_entropy - entropy

        return gain, thresholds

    def make_tree_cont(self, data, classes, feature_names):
        """
        Creates a tree dictionary from the dataset
        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature_names: Names of the features which are observed
        :return tree: A graph describing the constructed tree
        """

        # Find the number of features and data

        number_of_features = len(data[0])
        number_of_data = len(data)

        # If the dataset contains points of only a single class, we reached the leaf (return the class)

        if classes.count(classes[0]) == number_of_data:
            return classes[0]

        # If there is no more data, then return the most frequent class at the previous node

        elif number_of_data == 0 or number_of_features == 0:

            return classes[0] #HAVE TO FIX THIS

        else:
            tree = {}
            max_gain = 0
            selected_feature_index = 0
            for feature_index, feature in enumerate(feature_names):
                gain, thresholds = self.calc_info_gain_cont(data, classes, feature_index)
                max_gain_index = np.argmax(gain)
                if gain[max_gain_index] > max_gain:
                    max_gain = gain[max_gain_index]
                    threshold = thresholds[max_gain_index]
                    selected_feature_index = feature_index

            # Create new upper and lower data
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

            upper_subtree = self.make_tree_cont(upper_data, upper_classes, feature_names)
            lower_subtree = self.make_tree_cont(lower_data, lower_classes, feature_names)

            # When coming back up merge the trees

            tree[feature] = {'>'+str(threshold): upper_subtree, '<='+str(threshold): lower_subtree}

            return tree

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


            # Find the feature with the greatest information gain

    def make_tree(self, data, classes, feature_names):
        """
        Creates a tree dictionary from the dataset
        :param data: Dataset used to construct the tree
        :param classes: Classes that the data takes
        :param feature_names: Names of the features which are observed
        :return tree: A graph describing the constructed tree
        """

        # Find the number of features and data

        number_of_features = len(data[0])
        number_of_data = len(data)

        # Finding possible classes

        class_values = []
        for aclass in classes:
            if class_values.count(aclass) == 0:
                class_values.append(aclass)

        # Calculate total entropy
        frequency = np.zeros(len(class_values))
        total_entropy = 0
        for class_index, aclass in enumerate(class_values):
            frequency[class_index] = classes.count(aclass)
            total_entropy += self.calc_entropy(float(classes.count(aclass))/float(number_of_data))

        # If the dataset contains points of only a single class, we reached the leaf (return the class)

        if classes.count(classes[0]) == number_of_data:
            return classes[0]

        # If there is no more data, then return the most frequent class at the previous node

        elif number_of_data == 0 or number_of_features == 0:
            return classes[np.argmax(frequency)]

        else:
            gain = np.zeros(number_of_features)
            for feature_index in range(number_of_features):
                g = self.calc_info_gain(data, classes, feature_index)
                gain[feature_index] = total_entropy - g

            best_feature_index = np.argmax(gain)
            tree = {feature_names[best_feature_index]:{}}
            values = []

            # Take the datapoints that the feature value with the greatest information gain takes

            for datapoint in data:
                if values.count(datapoint[best_feature_index]) == 0:
                    values.append(datapoint[best_feature_index])

            # For each value of the selected feature remove the feature from the dataset and create a subtree

            for value in values:
                new_data = []
                new_classes = []

                # Extract the data which has the current feature value

                for data_index, datapoint in enumerate(data):
                    if datapoint[best_feature_index] == value:
                        new_datapoint = list(datapoint)
                        new_datapoint.pop(best_feature_index)
                        new_feature_names = list(feature_names)
                        new_feature_names.pop(best_feature_index)
                        new_data.append(new_datapoint)
                        new_classes.append(classes[data_index])

                # Create a new subtree out of remaining data

                subtree = self.make_tree(new_data, new_classes, new_feature_names)

                # When coming back up merge the trees

                tree[feature_names[best_feature_index]][value] = subtree

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
                print ind, "|", key
                self.print_tree(tree[key], ind+"\t")
        else:
            print ind, "->", tree

    def classify(self, tree, datapoint, feature_names):
        """
        Classifies the datapoint using the decision tree
        :param tree: Dictionary of a decision tree
        :param datapoint: Data point to be classified
        :param feature_names: Feature names of the data
        :return data_class: Class of the data point
        """

        if type(tree) == dict:
            for key in tree.keys():
                for feature_index, feature_name in enumerate(feature_names):
                    if feature_name == key:
                        data_class = self.classify(tree[key][datapoint[feature_index]], datapoint, feature_names)
        else:
            return tree

        return data_class
