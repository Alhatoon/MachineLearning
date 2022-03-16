import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics



df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df = df.dropna()

df.head()
Y = df['Chance of Admit ']
df = df.drop(['Chance of Admit ', 'Serial No.'], axis=1)

X = df.to_numpy()
y = Y.to_numpy().transpose()
m, n = X.shape

ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)

iter = 5000
alpha = 0.00001
theta = np.zeros((n + 1))


def computeCost(X, y, theta):
    m = len(y)
    diff = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(diff, diff)
    return J


def gradientDescent(X, y, theta, alpha, iter):
    m = len(y)
    J_history = []

    for i in range(iter):
        hc = np.matmul(X, theta) - y
        theta -= alpha / m * np.matmul(X.transpose(), hc)

        J_history.append(computeCost(X, y, theta))
    return theta, J_history



theta, J_history = gradientDescent(X, y, theta, alpha, iter)
plt.title(" Without Normalization and Alpha 0.00001")
plt.plot(np.arange(iter), J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print(theta)

# RMSE error rate
y_pred = np.matmul(X, theta)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y, y_pred)))