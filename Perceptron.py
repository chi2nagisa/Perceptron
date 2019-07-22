import numpy as np

# Data set
X = np.array([[1, 1, 0, 0],
              [1, 0, 1, 0]])
Y = np.array([[1, 0, 0, 0]])

m = X.shape[1]

def sigmoid(x):
    return 1/(1+np.exp(-x))


alpha = 0.01  # Learning rate


b = np.random.rand(1, 1)
w = np.random.rand(1, 2)

for iter in range(100000):
    # FP
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    cost = -1/4 * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1 - A).T))
    print(cost)

    # BP
    dZ = A - Y
    dw = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ)
    w = w - alpha * dw
    b = b - alpha * db


# check
Test_x = np.array([[0], [1]])
Test_y = sigmoid(np.dot(w, Test_x) + b)
print("For x=(0,1), Pridiction is{}".format(Test_y))
