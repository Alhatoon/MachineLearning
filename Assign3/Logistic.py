import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix

df = pd.read_csv('dataset.csv')
df = df.dropna()
np.seterr(divide='ignore', invalid='ignore')

X = df.iloc[:, 1:4]
y = df.iloc[:, -1]

admitted = df.loc[y == 1]
not_admitted = df.loc[y == 0]
theta = np.zeros((X.shape[1], 1))
y = y.to_numpy().transpose()
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)
y = y[:, np.newaxis]


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def computeLRCost(x, y, theta):
    h = sigmoid(np.dot(x, theta))
    cost = (-(1 / X.shape[0]) * (np.sum(y * np.log(h)))) + (1 - y) * np.log(1 - h)
    grad = np.dot(x.T, y - h) / X.shape[0]
    return cost, grad


def TrainLRModel(x, y):
    theta = np.zeros(x.shape[1])
    iter = 6000
    theta = np.expand_dims(theta, axis=-1)
    for i in range(iter):
        cost, grad = computeLRCost(x, y, theta)
        theta += (0.03 * grad)
    return theta


def predictClass(X, theta, threshold):
    g = sigmoid(np.dot(X, theta))
    y_pred = []
    for i in range(len(g)):
        if g[i] >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


def testPerformance(y, y_predicted):
    confusion_matrix(y, y_predicted)
    acc = accuracy_score(y, y_predicted)
    recall = recall_score(y, y_predicted, average=None)
    precision = precision_score(y, y_predicted, average=None, zero_division=0)
    f_Score = f1_score(y, y_predicted, average=None)
    print("Accuracy: {:.1f}%".format(acc * 100))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F Score:", f_Score)
    return acc, recall, precision, f_Score


X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=1)

theta = TrainLRModel(X_train, y_train)
thershold = sigmoid(0.5)
y_predicted = predictClass(X_test, theta, thershold)

acc, recall, precision, f_Score = testPerformance(y_test, y_predicted)
