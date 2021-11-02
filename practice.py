import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lng

data = pd.read_csv("./Advertising.csv")
X = data["TV"].values.reshape(-1, 1)
ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X], 1)
Y = data["sales"].values.reshape(-1, 1)

alpha = 0.0001

theta = np.array([0, 0])

def hyp (X, theta):
    m = X * theta
    h = (m[:, 0]).reshape(-1, 1) + (m[:, 1]).reshape(-1, 1)
    return h

def function_of_cost (X, Y, theta):
    sq_of_dif = np.power(hyp (X, theta) - Y, 2)
    print (np.sum (sq_of_dif) / (2 * len(sq_of_dif)))

def gradient_descent(alpha, theta, X, Y):
    h = hyp (X, theta)
    theta0 = (alpha/len (X)) * np.sum(h - Y)
    theta1 = (alpha/len (X)) * np.sum((h - Y) * X[:, 1].reshape(-1, 1))
    theta = theta - np.array([theta0, theta1])
    return theta


theta_ = gradient_descent (alpha, theta, X, Y)
X_ = ((X[:, 1]).reshape(-1, 1))
Y_ = hyp (X, theta_ )

plt.plot(X_, Y_)
plt.scatter(
    data['TV'],
    data['sales'],
    c = 'purple'
    )
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show






