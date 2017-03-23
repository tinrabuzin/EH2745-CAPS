import unittest
import Trees


class SimplisticTest(unittest.TestCase):

    def setUp(self):
        self.data = Trees.Tree.Data('Activity')
        self.data.read_data('test_data.csv', 'test_datatypes.csv')

    def test_read_data(self):
        self.assertEqual(self.data.attr_types, ['False', 'False', 'False', 'False'])
        self.assertEqual(self.data.attributes, ['Deadline', 'Party', 'Lazy', 'Activity'])
        self.assertEqual(self.data.examples, [['Urgent', 'Yes', 'Yes', 'Party'],
                                                ['Urgent', 'No', 'Yes', 'Study'],
                                                ['Near', 'Yes', 'Yes', 'Party'],
                                                ['None', 'Yes', 'No', 'Party'],
                                                ['None', 'No', 'Yes', 'Pub'],
                                                ['None', 'Yes', 'No', 'Party'],
                                                ['Near', 'No', 'No', 'Study'],
                                                ['Near', 'No', 'Yes', 'TV'],
                                                ['Near', 'Yes', 'Yes', 'Party'],
                                                ['Urgent', 'No', 'No', 'Study']])

    def test_count_ones(self):
        self.assertEqual(Trees.Tree.count_ones([[1, '1'], [2, '0'], [3, '1']], 1), 2)

    def test_class_values(self):
        self.assertEqual(self.data.find_class_values(), ['Party', 'Study', 'Party', 'Party', 'Pub', 'Party',
                                                         'Study', 'TV', 'Party', 'Study'])

    def test_preprocess_classes(self):
        self.data.examples[0][3] = '?'
        self.data.preprocess()
        self.assertEqual(self.data.examples[0][3], 'Party')

if __name__ == '__main__':
    unittest.main()