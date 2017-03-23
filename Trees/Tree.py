import numpy as np
from collections import Counter


class Data:
    """
    Class Data containing the processed data file
    """
    def __init__(self, classifier):
        """
        Constructor which initialises the data
        :param classifier: Name of the class column in the dataset
        """
        self.examples = []
        self.attributes = []
        self.attr_types = []
        self.class_index = None
        self.classifier = classifier

    def read_data(self, data_file, data_type):
        """
        Reads the data from a csv file and processes it
        :param data_file: Location of the data file
        :param data_type: Location of the file specifying data types
        """

        f = open(data_file)
        read_file = f.read()
        split_data = read_file.splitlines()

        # Creating a list of examples where each example is a list
        self.examples = [row.split(',') for row in split_data]

        # Remove the first row since this describes the data attributes
        self.attributes = self.examples.pop(0)

        f.close()

        f = open(data_type)
        read_file = f.read()
        self.attr_types = read_file.split(',')

    def find_class_values(self):
        """
        Finds class values from the data set and set the class_index in the Data class
        :return class_values: Returns all of the class values found in the dataset
        """

        for column_index, column_name in enumerate(self.attributes):
            if column_name == self.classifier:
                self.class_index = column_index
        if self.class_index is None:
            self.class_index = len(self.attributes)-1

        class_values = [example[self.class_index] for example in self.examples]

        return class_values

    def preprocess(self):
        """
        Preprocesses the data
        :return:
        """

        print "==> Preprocessing the dataset"

        class_values = self.find_class_values()

        # Find the most commond class appearing in the data
        class_mode = Counter(class_values)
        class_mode = class_mode.most_common(1)[0][0]

        for attr_index in range(len(self.attributes)):

            # Take elements that have class 0 and extract the current attribute values
            ex_0class = filter(lambda x: x[self.class_index] == '0', self.examples)
            attr_value_0class = [example[attr_index] for example in ex_0class]

            # Take elements that have class 0 and extract the current attribute values
            ex_1class = filter(lambda x: x[self.class_index] == '1', self.examples)
            attr_value_1class = [example[attr_index] for example in ex_0class]

            values = Counter(attr_value_0class)
            mode0 = values.most_common(1)[0][0] # Most common attribute value for 0 class
            if mode0 == '?':
                mode0 = values.most_common(2)[1][0]

            values = Counter(attr_value_1class)
            mode1 = values.most_common(1)[0][0] # Most common attribute value for 1 class
            if mode1 == '?':
                mode1 = values.most_common(2)[1][0]

            mode01 = [mode0, mode1]

            # Iterate through examples and replace missing values

            for example in self.examples:
                if example[attr_index] == '?':
                    if example[self.class_index] == 0:
                        example[attr_index] = mode0
                    elif example[self.class_index] == 1:
                        example[attr_index] = mode1
                    else:
                        example[self.class_index] = class_mode
                        example[attr_index] = mode01[int(class_mode)]

            for example in self.examples:
                if self.attr_types[attr_index] == 'True':
                    example[attr_index] = float(example[attr_index])


class Node():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None


def count_ones(examples, classifier_index):
    count = 0
    for example in examples:
        if example[classifier_index] ==  '1':
            count += 1
    return count


def calc_dataset_entropy(examples, class_index):
    """
    Calculates the entropy of a given dataset
    :param data: Data class describing a whole dataset
    :return entropy: The calculated entropy of a dataset
    """

    n_data = len(examples)
    ones = count_ones(examples, class_index)
    zeros = n_data - ones

    entropy = 0
    p = ones/n_data
    if p!=0:
        entropy += -p*np.log2(p)
    p = zeros/n_data
    if p!=0:
        entropy += -p*np.log2(p)

    return entropy


def calc_gain(dataset, entropy, val, attr_index):

    # Classifier for the split sets is now the attribute

    classifier = dataset.attributes[attr_index] # Attribute index according to witch the gain is calculated

    attr_entropy = 0
    n_examples = len(dataset.examples)

    # Create upper and lower datasets

    gain_upper_dataset = Data(classifier)
    gain_lower_dataset = Data(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attr_types = dataset.attr_types
    gain_lower_dataset.attr_types = dataset.attr_types

    for example in dataset.examples:
        if example[attr_index] >= val:
            gain_upper_dataset.examples.append(example)
        elif example[attr_index] < val:
            gain_lower_dataset.examples.append(example)

    # If we have a split that is invalid return -1

    if len(gain_lower_dataset) == 0 or len(gain_upper_dataset) == 0:
        return -1

    # Calculate entropy for each of the splits
    attr_entropy += len(gain_upper_dataset.examples)/n_examples*calc_dataset_entropy(gain_upper_dataset, attr_index)
    attr_entropy += len(gain_lower_dataset.examples)/n_examples*calc_dataset_entropy(gain_lower_dataset, attr_index)

    return entropy - attr_entropy

def classify_leaf(dataset, class_index):
    """
    Method that classifies leaves based on the maximum number of classes
    :param dataset:
    :param class_index:
    :return:
    """
    ones = count_ones(dataset, class_index)
    total = len(dataset.examples)
    zeros = total-ones
    if ones >= zeros:
        return 1
    else:
        return 0


def make_tree(dataset, parent_node, classifier):

    # Create a new node

    node = Node(True, None, None, parent_node, None, 0)

    # Assign the node height

    if parent_node is None:
        node.height = 1
    else:
        node.height = node.parent.height() + 1
    return 0

    ones = count_ones(dataset.examples, dataset.class_index)

    # Check if the dataset is pure and in that case return leafs

    if ones == len(dataset.examples):
        node.classification = 1
        node.is_leaf = True
        return node
    elif ones == 0:
        node.classification = 0
        node.is_leaf = True
        return node
    else:
        node.is_leaf = False

    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = None
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset.examples, dataset.class_index)  # Total dataset entropy

    # Determine the best attribute to split on and its split value

    for attr_index in range(len(dataset.attributes)):
        if dataset.attributes[attr_index] != classifier:
            local_max_gain = 0
            local_split_val = None

            # Find all of the possible values of attributes

            attr_values = [example[attr_index] for example in dataset.examples]
            attr_values = list(set(attr_values))

            # If we have a lot of values, reduce the possible combinations in steps of 10%

            if len(attr_values) > 100:
                attr_values = sorted(attr_values)
                total = len(attr_values)
                ten_precentile = int(total/10)
                new_list = []
                for mult_precentile in range(1,10):
                    new_list.append(attr_values[mult_precentile*ten_precentile])
                attr_values = new_list

            # Iterate through values to check which one is the best to split on

            for val in attr_value_list:

                local_gain = calc_gain(dataset, dataset_entropy, val, attr_index)

                if local_gain > local_max_gain:
                    local_max_gain = local_gain
                    local_split_val = val

            # On each iteration check if this is the best attribute to use for splitting

            if local_max_gain > max_gain:
                max_gain = local_max_gain
                split_val = val
                attr_to_split = attr_index

    assert split_val is None or attr_to_split is None, "Could not find the best split"

    if max_gain <= min_gain or node.height > 20:
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)
        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_index]

if False:

    data = Data(None)

    data.read_data("/Users/tinrabuzin/dev/EH2745-CAPS/Trees/btrain.csv", "/Users/tinrabuzin/dev/EH2745-CAPS/Trees/datatypes.csv")

    data.preprocess()