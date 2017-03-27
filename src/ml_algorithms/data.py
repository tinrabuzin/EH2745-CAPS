
class Data():

    def __init__(self, data=None, classes=None, feature_names=None, feature_types=None):
        """
        Constructor of a class data

        :param data: Array of data point arrays with its attribute values
        :param classes: Classes assigned to each data point
        :param feature_names: Names of the attributes
        :param feature_types: Types of the atributes (numeric or not)
        """

        self.data = data
        self.classes = classes
        self.feature_names = feature_names
        self.feature_types = feature_types

    def read_data(self, file_name):
        """
        Reads the data from a *.csv file and stores it in the data objects. Also, returns the read information.

        :param file_name: Path to the dataset
        :return data: The data read from a file (values of the attributes)
        :return classes: The classes that each of the examples takes
        :return feature_names: Names of the attributes describing examples
        """

        fid = open(file_name, 'r')
        self.data = []
        self.classes = []
        self.feature_names = []
        for line_index, line in enumerate(fid.readlines()):
            line = line.strip()
            if line_index == 0:
                self.feature_names = line.split(',')[:-1]
            else:
                self.data.append(map(float, line.split(',')[:-1]))
                if float(line.split(',')[-1]) > 0:
                    self.classes.append(1)
                else:
                    self.classes.append(0)

        fid.close()

        return self.data, self.classes, self.feature_names

