import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4)

# Split the 40% subset above into two: one half for validation and the other for the test set
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.5)

del X_, y_


# Record best k accuracy
best_k_avg_acc = 0
best_k = 0
    
# Iterate over values of k from 3 to 11
for k in range(3, 12):
    print("K = %d" % k)

    # Record validation accuracy
    avg_acc = 0
    
    # Repeat the process 10 times to get a stable estimate
    for j in range(10):
        
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Train the model
        knn.fit(X_train, y_train)
        
        # Model prediction
        acc = knn.score(X_val, y_val)
        avg_acc += acc
        # print("Acc = %.2f" % acc)
    
    # Average accuracy for the current k
    avg_acc /= 10
    print("Average ACC = %.2f" % avg_acc)
    
    # Check if this k is the best
    if avg_acc > best_k_avg_acc:
        best_k_avg_acc = avg_acc
        best_k = k

print("Best k: %d, Average validation accuracy for best k: %.2f" % (best_k, best_k_avg_acc))


# Record test accuracy
avg_acc = 0

for i in range(10):
    # Train the final model using the best k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # Model prediction on test set
    test_acc = knn.score(X_test, y_test)
    print("Test accuracy for iteration %d: %.2f" % (i+1, test_acc))
    avg_acc += test_acc

# Average accuracy across iterations
avg_acc /= 10
print("Average test accuracy: %.2f" % avg_acc)
