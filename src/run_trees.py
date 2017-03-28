import ml_algorithms
import matplotlib.pyplot as plt
import random


if True:
    tree = ml_algorithms.BasicTree()
    data, classes, feature_names = tree.read_data('/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv')
    feature_types = ['Numeric', 'Numeric']

    training_data, training_classes, test_data, test_classes = tree.create_training_test_sets(data, classes, 0.6)

    #x = [point[0] for point in dataset]
    #y = [point[1] for point in dataset]
    #plt.scatter(x, y, s=20, c = classes)
    #plt.gray()
    #plt.draw()
    #plt.show()

    made_tree = tree.make_tree(training_data, training_classes, feature_names, feature_types)
    tree.print_tree(made_tree)
    perc = tree.tree_test(made_tree, test_data, test_classes, feature_names)
    print perc


else:



    dataset = [[10,10], [30,40], [60,70], [80,90],[20,80],[30,70],[70,10],[90,20]]
    classes = [1, 1, 1, 1,0,0,0,0]
    feature_names = ['f1', 'f2']

    dtree = ml_algorithms.BasicTree()
    data = ml_algorithms.Data(dataset, classes, feature_names, ['NotNumeric', 'NotNumeric'] )

    constructed_tree = dtree.make_tree(dataset, classes, feature_names, ['NotNumeric', 'NotNumeric'] )

    dtree.print_tree(constructed_tree)
