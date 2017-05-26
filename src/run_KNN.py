import ml_algorithms
import matplotlib.pyplot as plt

# =============================================================================================
"""
This run_KNN scripti imports data from "dataset.csv", calculates the prediction
accuracy for all different vlaues of k and prints the results.

In the initailization part we can choose to normalize the data and the value of k that we want to test.
The three most accurate k's are printed out and all k's are plotted.
"""

# =============================================================================================
#					Part 0 - Initialize choices
# =============================================================================================

# within the method "correct_classes_for_k" you can normalize the X-vules if norm=1
norm = 1
# choice of number of tested k's
k = 3
# =============================================================================================
#					Part I - IMPORT DATA
# =============================================================================================

# define column names
file_name = '/Users/marnil3/Documents/Python Scripts/EH2745-CAPS-master/EH2745-CAPS-master/src/datasets/dataset.csv'

# loading training data, using the method Tin created, however I changed from map to list (data2)
data_obj = ml_algorithms.Data(file_name)
data, classes, feature_names = data_obj.read_data(file_name)

# Test with less data len(data)
end_data = len(data)
data = data[:end_data]
classes = classes[:end_data]


# =============================================================================================
#					Part II - calculate the accuracy for all values of k
# =============================================================================================
#   1. Normalize data
#   2a. Calculate distances for all objects
#   2b. Predict class for all objects depending on NN
#   3. Do 2a and b for all values of k and calculate amount of correct classes
#   4. Find the best predictions

#   1. Normalize the data by substracting mean and devide by std
if norm == 1:
    data = normalize(data)

DISTANCES = []
DATA2 = []
#   2a. Calculate distances
for i in range(len(data)):
    # get one object (o_i) and
    o_i, data2 = KNN.split_data(data, i)
    distances = KNN.calc_distances(data2, o_i)
    DISTANCES.append(distances)
    DATA2.append(data2)

# create a list of odd k-values
K = list(range(3, k, 2))

# lists to store results
correct_for_K = []
corr_plot_percent = []
corr_plot_K = []

#   3. Calculate the accuracy for all k-values
for k in K:
    # store predictions for each k-value
    Y_pred = []
    #   2b. Classify all objects
    for i in range(len(data)):
        class_pred = KNN.classify_KNN(DISTANCES[i], DATA2[i], k)
        Y_pred.append(class_pred)

    correct_classes_for_k = KNN.calc_correct(data, Y_pred)

    # Store values
    correct_for_K.append([correct_classes_for_k, k])
    corr_plot_percent.append([correct_classes_for_k])
    corr_plot_K.append([k])

#   4. Find the best k-values
correct_for_K_sorted = sorted(correct_for_K)

# =============================================================================================
#					Part III - print and plot the results
# =============================================================================================

# print the best three k values
print ("The best accuracy is:           ",
       correct_for_K_sorted[-1][0], ",achieved with k:", correct_for_K_sorted[-1][1])
print ("The second accuracy is:         ",
       correct_for_K_sorted[-2][0], ",achieved with k:", correct_for_K_sorted[-2][1])
print ("The third accuracy is:          ",
       correct_for_K_sorted[-3][0], ",achieved with k::", correct_for_K_sorted[-3][1])


# plot accuracy and and k values
plt.plot(corr_plot_K, corr_plot_percent)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.show()
