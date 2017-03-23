import Trees

data = Trees.Data.Data('DecisionTree')

data.read_data("/Users/tinrabuzin/dev/EH2745-CAPS/Trees/btrain.csv", "/Users/tinrabuzin/dev/EH2745-CAPS/Trees/datatypes.csv")

data.preprocess()

#x = Trees.decision_tree.BasicTree()

#data, classes, feature_names =  x.read_data('/Users/tinrabuzin/dev/EH2745-CAPS/Trees/dtree_dataset.csv')

#data = [[10,10], [30,40], [60,70], [80,90],[20,80],[30,70],[70,10],[90,20]]
#classes = [1, 1, 0, 0,1,1,0,0]


#tree = x.make_tree_cont(data, classes, ['feature1','feature2'])

#x.print_tree(tree)
#p = x.calc_entropy(3)
#tree = x.make_tree(x.data, x.classes, x.features)
#x.print_tree(tree)
#print x.classify(tree, ['Urgent', 'No', 'Yes'], ['Deadline', 'Party', 'Lazy'])
