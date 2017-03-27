
class Data():

    def __init__(self, dataset=None, classes=None, feature_names=None, feature_types=None):
        """
        Constructor of a class data

        :param dataset: Array of data point arrays with its attribute values
        :param classes: Classes assigned to each data point
        :param feature_names: Names of the attributes
        :param feature_types: Types of the atributes (numeric or not)
        """

        self.dataset = dataset
        self.classes = classes
        self.feature_names = feature_names
        self.feature_types = feature_types

    def read_data(self, file_name):
        """
        Reads the data from a *.csv file and stores it in the data objects. Also, returns the read information.

        :param file_name: Path to the dataset
        :returns data: The data read from a file (values of the attributes)
        :returns classes: The classes that each of the examples takes
        :returns feature_names: Names of the attributes describing examples
        """

        fid = open(file_name, 'r')
        self.dataset = []
        self.classes = []
        self.feature_names = []
        for line_index, line in enumerate(fid.readlines()):
            line = line.strip()
            if line_index == 0:
                self.feature_names = line.split(',')[:-1]
            else:
                self.dataset.append(map(float, line.split(',')[:-1]))
                if float(line.split(',')[-1]) > 0:
                    self.classes.append(1)
                else:
                    self.classes.append(0)

        fid.close()

        return self.dataset, self.classes, self.feature_names

    def split_data_cont(self, threshold, feature_index):
        """
        Splits the data object according to the provided threshold of a selected attribute

        :param threshold:
        :param feature_index:
        :return:
        """

        lower_data = Data()
        upper_data = Data()
        for data_index, datapoint in enumerate(self.dataset):
                if datapoint[feature_index] <= threshold:
                    lower_data.dataset.append(datapoint)
                    lower_data.classes.append(self.classes[data_index])
                elif datapoint[feature_index] > threshold:
                    upper_data.dataset.append(datapoint)
                    upper_data.classes.append(self.classes[data_index])

        return lower_data, upper_data
