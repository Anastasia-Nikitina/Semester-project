import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lng

data = pd.read_csv("./dataset.csv")

X = data.iloc[:, 0]
Y = data.iloc[:, 1]

theta_0 = 1
theta_1 = 1
alpha = 0.0001
n = len(X)
epsilon = 0.0001

def hyp(X, theta_0, theta_1):
    return(X*theta_1 + theta_0)

H = hyp(X, theta_0, theta_1)

while True:
    H = hyp(X, theta_0, theta_1)
    theta_0 -= (alpha/n) * np.sum(H - Y)
    theta_1 -= (alpha/n) * np.sum(X * (H - Y))
    if all (np.absolute(H - hyp(X, theta_0, theta_1)) < epsilon):
        break
    
    
Y_ = hyp(X, theta_0, theta_1) 
plt.plot([min(X), max(X)], [min(X)*theta_1 + theta_0, max(X)*theta_1 + theta_0], color='blue')
plt.scatter(X, Y, c = 'purple')
plt.show



