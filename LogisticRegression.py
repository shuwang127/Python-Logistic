# read data from .txt file.
import numpy as np
f = open('Student-Pass-Fail-Data.txt', 'r')
next(f) # skip the first row.
data = np.array([[int(num) for num in line.split(',')] for line in f])
print('data:\n', data)

# divide data into features (Self_Study_Daily, Tuition_Monthly) and labels (Pass_Or_Fail).
x = data[:,:-1]
y = data[:,-1]
print('x:\n', x, '\ny:\n', y)

# split the data into train and test sets.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# define the Logistic Regression model.
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()

# fit the training data with LR model. (take two parameters: x_train and y_train)
logistic_regression.fit(x_train, y_train)

# get predictions for testing data.
y_pred = logistic_regression.predict(x_test)
print(y_pred)

# calculate the testing accuracy with actual labels and predicted labels.
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Testing accuracy = %.2f%%.' % (accuracy * 100))

# plot the decision boundary visually.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x1, x2 = np.meshgrid(np.arange(min(x[:,0])-1, max(x[:,0])+1, 0.02), np.arange(min(x[:,1])-1, max(x[:,1])+1, 0.02)) # mesh.
Z = logistic_regression.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape) # get predictions for mesh.
plt.contourf(x1, x2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green'))) # plot decision boundary.
for cls in range(2):
    plt.scatter(x=x_train[y_train==cls, 0], y=x_train[y_train==cls, 1], alpha=0.5, c=('red', 'green')[cls], marker=('x', '^')[cls], label=cls) # plot training set.
    plt.scatter(x=x_test[y_test == cls, 0], y=x_test[y_test == cls, 1], alpha=0.8, c=('red', 'green')[cls], marker=('o', 's')[cls], label=cls) # plot testing set.
plt.show()
