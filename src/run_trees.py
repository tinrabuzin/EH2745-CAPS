import ml_algorithms


file_name_data = '/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv'
file_name_tcc = '/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset_tcc.csv'
data_obj = ml_algorithms.Data()
data_obj.read_data(file_name_data)
data_obj.read_tcc(file_name_tcc)


training_data, training_classes, test_data, test_classes = data_obj.create_training_test_sets(
    data_obj.data, data_obj.classes, 0.6)

tree = ml_algorithms.BasicTree()

made_tree = tree.make_tree(training_data, training_classes, data_obj.feature_names)

tree.print_tree(made_tree)

print tree.classify(made_tree, [60, 10], data_obj.feature_names)

perc, result_classes = tree.tree_test(made_tree, test_data, test_classes, data_obj.feature_names)
print "Percentage of the correctly classified states is: %f" % perc

tree.plot_results(training_data, training_classes, test_data, test_classes, result_classes)

