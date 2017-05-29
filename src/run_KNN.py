import ml_algorithms
import numpy as np
import matplotlib.pyplot as plt

file_name = '/Users/tinrabuzin/dev/EH2745-CAPS/src/datasets/dataset.csv'

# Load the data set from a file
data_obj = ml_algorithms.Data(file_name)
data, classes, feature_names = data_obj.read_data()

# Normalise the data
if False:
    data = data_obj.normalize(data)

# Splitting the data into training and test data
training_data, training_classes, test_data, test_classes = data_obj.create_training_test_sets(
    data, classes, 0.7)

# Instantiate the k-NN object
knn = ml_algorithms.KNN()

# Iterate through different numbers of k and estimate classification accuracy for the test set
k_dict = {}
for k in [3, 6, 9, 11, 13]:
    print "----- Classifying for k = %d -----" % (k)

    est_classes = []
    for index in range(len(test_data)):
        distances = knn.neighbours_distance(training_data, test_data[index])
        est_classes.append(knn.classify(3, distances, training_classes))

    # Calculate the accuracy
    correct_num = np.sum(np.array(est_classes) == np.array(test_classes))
    k_dict[k] = float(correct_num) / len(test_classes)
    print "Accuracy is: %d" % (k_dict[k])
    k_dict_sort = sorted(k_dict.iteritems(), key=lambda (k, v): (v, k))

# Print the best three k values
print ("The best accuracy is: ",
       k_dict_sort[0][1], ",achieved with k - ", k_dict_sort[0][0])
print ("The second best accuracy is:         ",
       k_dict_sort[1][1], ",achieved with k -", k_dict_sort[1][0])
print ("The third best accuracy is:          ",
       k_dict_sort[2][1], ",achieved with k -", k_dict_sort[2][0])


# Plot accuracy vs number of neighbours

fig, ax = plt.subplots()
ax.scatter(k_dict.keys(), k_dict.values())
ax.set_xlabel('Number of Neighbors K')
ax.set_ylabel('Accuracy')
plt.show()
