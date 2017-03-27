import ml_algorithms
import matplotlib.pyplot as plt
import random


if True:
    data = ml_algorithms.Data()
    data.read_data('/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv')
    data.feature_types = ['Numeric', 'Numeric']
    dataset, classes = zip(*random.sample(list(zip(data.dataset, data.classes)), int(0.6*len(data.dataset))))

    #x = [point[0] for point in dataset]
    #y = [point[1] for point in dataset]
    #plt.scatter(x, y, s=20, c = classes)
    #plt.gray()
    #plt.draw()
    #plt.show()

    tree = ml_algorithms.BasicTree()
    made_tree = tree.make_tree(dataset, classes, data.feature_names, data.feature_types)
    tree.print_tree(made_tree)

else:



    dataset = [[10,10], [30,40], [60,70], [80,90],[20,80],[30,70],[70,10],[90,20]]
    classes = [1, 1, 1, 1,0,0,0,0]
    feature_names = ['f1', 'f2']

    dtree = ml_algorithms.BasicTree()
    data = ml_algorithms.Data(dataset, classes, feature_names, ['NotNumeric', 'NotNumeric'] )

    constructed_tree = dtree.make_tree(dataset, classes, feature_names, ['NotNumeric', 'NotNumeric'] )

    dtree.print_tree(constructed_tree)
