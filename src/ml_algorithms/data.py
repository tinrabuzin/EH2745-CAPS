import random


class Data:

    def __init__(self):
        """
        Constructor of a data object
        """

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

        self.data = data
        self.classes = classes
        self.feture_names = feature_names

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

    def normalize(self, data):
        """
        Normalize X

        .. math::
            (X-\mu(X))/\sigma(X)

        :param x1:
        :param x2:
        :return: normalized X in data
        """
        # initialize
        X1 = 0
        X2 = 0
        tot = len(data)
        data2 = []
        # calculate mean
        for n in range(len(data)):
            x1 = float(data[n][0])
            x2 = float(data[n][1])
            X1 = X1 + x1
            X2 = X2 + x2

        X1_mean = X1 / tot
        X2_mean = X2 / tot

        # caclulate std
        for n in range(len(data)):
            x1 = np.square(float(data[n][0]) - X1_mean)
            x2 = np.square(float(data[n][1]) - X2_mean)
            X1 = X1 + x1
            X2 = X2 + x2

            X1_std = np.sqrt(X1 / tot)
            X2_std = np.sqrt(X2 / tot)

        # normalize data
        for n in range(len(data)):
            x1 = (float(data[n][0]) - X1_mean) / X1_std
            x2 = (float(data[n][1]) - X2_mean) / X2_std
            cl = int(data[n][2])
            data2.append([x1, x2, cl])

        return data2
