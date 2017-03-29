import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ml_algorithms import dtree


class AlgorithmsTestCases(unittest.TestCase):
    def setUp(self):
        self.tree = dtree.BasicTree()

    def test_calc_entropy(self):
        self.assertEqual(self.tree.calc_entropy(0.5), 0.5)
        self.assertEqual(self.tree.calc_entropy(0), 0)
        self.assertEqual(self.tree.calc_entropy(1), 0)


if __name__=='__main__':
    unittest.main()