#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 4
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg

myfile = open('hw4_problem4_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 1.0


#-------- Your code (~10 lines) ---------
n = X_train.shape[0]
K = np.zeros((n+1, n+1))

for i in range(n):
    for j in range(n):
        K[i][j] = sigma_f**2 * np.exp(-1/(2*ls**2) * (X_train[i] - X_train[j])**2) + sigma**2 * (i == j)

for idx in range(X_test.shape[0]):
    for i in range(n):
        K[i][n] = sigma_f**2 * np.exp(-1/(2*ls**2) * (X_train[i] - X_test[idx])**2) + sigma**2 * (i == n)
        K[n][i] = sigma_f**2 * np.exp(-1/(2*ls**2) * (X_train[i] - X_test[idx])**2) + sigma**2 * (i == n)
    K[n][n] = sigma_f**2 * np.exp(-1/(2*ls**2) * (X_test[idx] - X_test[idx])**2) + sigma**2
    predictive_mean[idx] = np.dot(np.matmul(K[n, :n], linalg.inv(K[:n, :n] + sigma ** 2 * np.identity(n))), Y_train)
    predictive_std[idx] = np.sqrt(K[n][n] - np.dot(np.matmul(K[n][:n], linalg.inv(K[:n, :n] + np.identity(n) * (sigma ** 2))), K[:n, n]))

    
#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=5, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
fig.set_size_inches((12, 8))
fig.tight_layout()
plt.savefig(f"./report/figures/sig={sigma}_sigf={sigma_f}_ls={ls}.png", dpi=300)
