import matplotlib.pyplot as plt
import pickle
import numpy as np


# Compute the sigmoid of z
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


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


d = pickle.load(open("visualize_variables/d", 'rb'))
test_set_x = pickle.load(open("visualize_variables/test_set_x", 'rb'))
test_set_y = pickle.load(open("visualize_variables/test_set_y", 'rb'))
num_px = pickle.load(open("visualize_variables/num_px", 'rb'))

# See what the model predicted an image was
# y = 1 -> the image is a cat, y = 0 -> the image is not a cat
# Change index to any value between 0 and 49 (inclusive)
index = 49
decision = "cat"
if d["Y_prediction_test"][0][index] == 0:
    decision = "non-cat"
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_set_y[0, index]) + ", model predicted that it is a " + decision + " picture.")
plt.show()
