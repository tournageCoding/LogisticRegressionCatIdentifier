import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import pickle
from lr_utils import load_dataset


# Compute the sigmoid of z
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# Initialise w and b with zeros
def initialise_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


# Complete the forward and backward propagation of w and b
def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward prop
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))) / m

    # Back prop
    dw = (np.dot(X, (A - Y).T)) / m
    db = np.sum(A - Y) / m

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# Update w and b using gradient descent
def update_parameters(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # Record the value of the cost function every 100 iterations
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration " + str(i) + ": " + str(cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# Predict whether input is 0 (non-cat) or 1 (cat) based on learned logistic regression parameters (w, b)
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute predicted probability of a cat being in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    # Convert probabilities to actual predictions
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.05, print_cost=False):
    w, b = initialise_with_zeros(X_train.shape[0])
    parameters, grads, costs = update_parameters(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: " + str(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: " + str(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# load in the data set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# initialise the variables for number of training examples, number of test examples and number of pixels
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape input so that each column is now one training example
# Eg train_set_x_flatten has shape (12288, 209)
# Images are 64x64 pixels and there are 3 layers of this for RGB. 64*64*3 = 12288. There are 209 training examples.
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# Standardise data to have values between 0 and 1
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Save variables to use trained model in other files
pickle.dump(d, open("visualize_variables/d", 'wb'))
pickle.dump(test_set_x, open("visualize_variables/test_set_x", 'wb'))
pickle.dump(test_set_y, open("visualize_variables/test_set_y", 'wb'))
pickle.dump(num_px, open("visualize_variables/num_px", 'wb'))
