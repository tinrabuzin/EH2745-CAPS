import ml_algorithms
import matplotlib.pyplot as plt
import random

data = ml_algorithms.Data()
data.read_data('/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv')

print data.feature_names

#tree = ml_algorithms.BasicTree()




#data = [[10,10], [30,40], [60,70], [80,90],[20,80],[30,70],[70,10],[90,20]]
#classes = [1, 1, 1, 1,0,0,0,0]
#feature_names = ['f1', 'f2']

data, classes = zip(*random.sample(list(zip(data,classes)),int(0.6*len(data))))

#x = [point[0] for point in data]
#y = [point[1] for point in data]

#plt.scatter(x, y, s=20, c = classes)
#plt.gray()
#plt.draw()

made_tree = tree.make_tree(data, classes, feature_names, ['Numerical','Numerical'])

tree.print_tree(made_tree)
#plt.show()
#p = x.calc_entropy(3)
#tree = x.make_tree(x.data, x.classes, x.features)
#x.print_tree(tree)
#print x.classify(tree, ['Urgent', 'No', 'Yes'], ['Deadline', 'Party', 'Lazy'])
