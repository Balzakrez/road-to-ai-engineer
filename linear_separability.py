import scipy
import numpy as np
from sklearn.datasets import load_iris

def check_linear_separability(X, Y):
    A = Y[:, np.newaxis] * X # righe: y_i * x_
    w = np.zeros(X.shape[1]) # w
    v = np.ones(X.shape[0])
    res = scipy.optimize.linprog(w, A_ub=-A, b_ub=-v, bounds=(None, None))
    if res.status != 0:
        print("Training set non linearmente separabile \n")
    else:
        w = res.x
        print(f"Training set linearmente separabile: w = {w.tolist()}")
        print("Tutti classificati correttamente:", np.all(np.sign(X.dot(w)) == Y), "\n")


iris_dataset = load_iris()
ones_column = np.ones(100)[:, np.newaxis]  # colonna di 1: notazione omogenea (bias)

# setosa vs virginica: separabili
X1 = np.concatenate((ones_column, np.concatenate((iris_dataset.data[:50], iris_dataset.data[100:]))),axis=1,)
Y1 = (np.concatenate((iris_dataset.target[:50], iris_dataset.target[100:])) - 1.0)  # {0,2} -> {-1,+1}

# setosa vs versicolor: separabili
X2 = np.concatenate((ones_column, iris_dataset.data[:100]), axis=1)
Y2 = (iris_dataset.target[:100] - 0.5) / 0.5  # etichette {0,1} -> {-1,+1}

# versicolor vs virginica: NON separabili
X3 = np.concatenate((ones_column, iris_dataset.data[50:]), axis=1)
Y3 = (iris_dataset.target[50:] - 1.5) / 0.5  # etichette {1,2} -> {-1,+1}

check_linear_separability(X1, Y1)
check_linear_separability(X2, Y2)
check_linear_separability(X3, Y3)