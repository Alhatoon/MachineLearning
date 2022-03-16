import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# MSE cost function
def calculateCost(X, y, theta):
    J = 0
    m=len(y)
    s = np.power((X.dot(theta) - np.transpose([y])), 2)
    J = (1.0 / (2 * m)) * s.sum(axis=0)
    return J


# gradient descent function
def gradientDescent(X, y, theta):
    # set the learning rate
    alpha = 0.0005
    # set the number of steps taken by gradient descent
    itarations = 4000

    # ====================== YOUR CODE HERE ======================
    m = y.shape[0]

    theta = theta.copy()

    Jhis = []

    for i in range(itarations):
        temp0 = theta[0] - alpha * (1 / m) * np.sum(((X[:, 0] * theta[0] + X[:, 1] * theta[1]) - y) * X[:, 0])
        temp1 = theta[1] - alpha * (1 / m) * np.sum(((X[:, 0] * theta[0] + X[:, 1] * theta[1]) - y) * X[:, 1])
    theta[0] = temp0
    theta[1] = temp1

    Jhis.append(calculateCost(X, y, theta))
    return theta


# ************************ regression script************************
# dataset loading
# training dataset
train_data = pd.read_csv("train.csv")
# removing Nan if any in dataset
train_data = train_data.dropna()

# **************** loading the data in numpy matrix ***********
X = train_data.iloc[:, 0].values
y = train_data.iloc[:, 1].values

# **************** ploting the dataset *********************
plt.scatter(X, y, marker='x')
plt.title(" training dataset ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# **************** Gradient descent **********************

ones = [1 for i in range(len(y))]
X = np.column_stack((ones, X))

# func gradient 
theta = gradientDescent(X, y, np.array([0, 0]))
print('theta:', theta)

# ****************** plot the model ***********************
t = np.dot(X, theta)
plt.scatter(X[:, 1], y, marker='x')
plt.plot(X[:, 1], t, color="red")
plt.show()

# ****************** predection ***************************
p1 = t = np.dot([1, 56], theta)
print('The predicted value for p1 is ', p1)
