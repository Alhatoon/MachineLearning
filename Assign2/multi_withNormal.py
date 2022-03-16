import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.head()
Y = df['Chance of Admit ']
df = df.drop(['Chance of Admit ', 'Serial No.'], axis=1)

X = df.to_numpy()
y = Y.to_numpy().transpose()
m, n = X.shape
noramlize_X = preprocessing.normalize(X)

ones = np.ones([noramlize_X.shape[0], 1])
noramlize_X = np.concatenate((ones, noramlize_X), axis=1)

iters = 1000
alpha = 0.1
theta = np.zeros((n + 1))


def computeCost(X, y, theta):
    m = len(y)
    diff2 = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(diff2, diff2)
    return J


def gradientDescent(X, y, theta, alpha, iters):
    m = len(y)
    J_history = []

    for i in range(iters):
        hc = np.matmul(X, theta) - y
        theta -= alpha / m * np.matmul(X.transpose(), hc)

        J_history.append(computeCost(X, y, theta))
    return theta, J_history


theta, J_history = gradientDescent(noramlize_X, y, theta, alpha, iters)
plt.title(" With Normalization and Alpha 0.1")
plt.plot(np.arange(iters), J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
print(theta)

# RMSE error rate
y_pred = np.matmul(noramlize_X, theta)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y, y_pred)))
