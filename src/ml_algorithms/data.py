import random
import numpy as np


class Data:

    def __init__(self, data=np.array([]), classes=np.array([]), tcc=np.array([]), feature_names=np.array([])):
        """
        Constructor of a data object
        """
        self.data = data
        self.classes = classes
        self.tcc = tcc
        self.feature_names = feature_names

    def read_tcc(self, file_name=None):

        fid = open(file_name, 'r')
        tcc = []
        for line_index, line in enumerate(fid.readlines()):
            if line_index != 0:
                tcc.append([float(line.strip())])
        fid.close()

        self.tcc = np.array(tcc)

        return self.tcc

    def read_data(self, file_name):
        """
        Reads the data from a *.csv file and stores it in the data objects.
        Also, returns the read information.

        Changes to Tins mathod: I have added data2 to use list instead of map..[:-1]

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

        self.data = np.array(data)
        self.classes = np.array(classes)
        self.feature_names = np.array(feature_names)

        return data, classes, feature_names

    def create_training_test_sets(self, data, classes, percentage):
        """
        Splits the data set into training and test sets

        :param data:
        :param classes:
        :return:
        """

        training_indices = random.sample(range(len(data)), int(percentage *
                                                               len(data)))
        test_indices = []
        for index in range(len(data)):
            if training_indices.count(index) == 0:
                test_indices.append(index)

        self.training_data = []
        self.training_classes = []
        self.test_data = []
        self.test_classes = []

        for index in training_indices:
            self.training_data.append(data[index])
            self.training_classes.append(classes[index])

        for index in test_indices:
            self.test_data.append(data[index])
            self.test_classes.append(classes[index])

        return self.training_data, self.training_classes, self.test_data, self.test_classes

    def normalize(self):
        """
        Normalises the data set

        .. math::
            (X-\mu(X))/\sigma(X)

        :return: normalized X in data
        """

        self.data = (self.data - self.data.mean()) / self.data.var()
        self.tcc = (self.tcc - self.tcc.mean()) / self.tcc.var()

        return self.data, self.tcc
