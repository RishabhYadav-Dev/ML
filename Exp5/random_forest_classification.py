# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# # Theoy
# Aim:
# To study and implement the Random Forest Classifier algorithm in Python for supervised classification.

# Objective:
# To understand how ensemble decision trees reduce overfitting and improve predictive accuracy through bootstrap aggregation (bagging) and random feature selection.

# Theory:
# Random Forest is a popular supervised machine learning algorithm used for both classification and regression tasks. It is an ensemble learning method based on Bagging (Bootstrap Aggregating) that constructs a multitude of individual Decision Trees during training and combines their outputs to produce a more accurate and stable prediction.

# While a single Decision Tree tends to suffer from high variance and overfitting to the training data, Random Forest overcomes this limitation by introducing two levels of randomness:
# 1. Bootstrap Sampling (Row Sampling): Each tree in the forest is trained on a random subset of the training data sampled with replacement (bootstrap sample).
# 2. Feature Randomness (Column Sampling): At each split node in a tree, instead of searching for the best feature among all available features, the algorithm selects a random subset of features and chooses the optimal split only from that subset.

# This decorrelates the individual trees in the forest. For classification tasks, the final output is determined by Majority Voting across all trees in the ensemble. For regression tasks, it takes the average of the predictions from all trees.

# Algorithm:
# 1. Load the dataset and separate the independent features (X) and target class labels (y).
# 2. Split the dataset into training and testing subsets.
# 3. Choose the number of trees (n_estimators) to build in the forest.
# 4. For each tree t = 1 to n_estimators:
#    - Draw a random bootstrap sample of size N with replacement from the training data.
#    - Grow a Decision Tree from the bootstrap sample by repeating the following steps at each node until a stopping criterion (such as max_depth) is reached:
#      a. Randomly select m features out of the total M features (typically m = sqrt(M) for classification).
#      b. Pick the best feature and split-point among the m selected features using Gini Impurity or Information Gain.
#      c. Split the node into two child nodes.
# 5. Aggregate the predictions from all trained Decision Trees:
#    - For classification: Output the class label with the highest number of votes (Majority Voting).
# 6. Evaluate the model performance on the test set using metrics such as Confusion Matrix and Accuracy Score.

# Formulas:
# 1. Gini Impurity (used to measure node purity and select splits):
#    Gini = 1 - sum_{i=1}^{C} (p_i)^2
#    (where p_i is the probability of an instance belonging to class i)

# 2. Entropy and Information Gain (alternative splitting criterion):
#    Entropy(S) = - sum_{i=1}^{C} p_i * log2(p_i)
#    Information Gain(S, A) = Entropy(S) - sum_{v in Values(A)} (|S_v| / |S|) * Entropy(S_v)

# 3. Majority Voting (Final Classification Prediction):
#    y_hat = mode { h_1(x), h_2(x), ..., h_T(x) }
#    (where h_t(x) is the prediction of the t-th decision tree)

# 4. Feature Subset Size per Split (Classification):
#    m = sqrt(M)
#    (where M is the total number of features)

# Advantages of Random Forest:
# - Significantly reduces overfitting compared to a single Decision Tree.
# - Handles high-dimensional datasets and missing values effectively.
# - Works well with both categorical and numerical features without requiring strict feature scaling.
# - Provides estimates of feature importance automatically.

# Conclusion:
# Hence, we have successfully studied the working principles, algorithm, and mathematical formulation of the Random Forest Classifier.