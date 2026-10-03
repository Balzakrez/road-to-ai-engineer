from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import numpy as np

def run_perceptron(X,Y):
    classifier = Perceptron(tol=1e-3, random_state=0, shuffle=False)
    classifier.fit(X,Y)
    print(f"\nAccuracy sul training set: {classifier.score(X, Y):.4f}, n_iter: {classifier.n_iter_}")

    # L'ordine dei dati influenza il risultato?
    indexes = np.arange(X.shape[0])
    scores = []
    iters = []
    for i in range(10):
        np.random.shuffle(indexes)
        classifier.fit(X[indexes, :], Y[indexes])
        score = classifier.score(X, Y)
        n_iter = classifier.n_iter_
        print(f"[{i}] Accuracy sul training set: {score:.4f}, n_iter: {n_iter}")
        scores.append(score)
        iters.append(n_iter)
    print(f"Accuracy media sul training set: {np.mean(scores):.4f} e con numero iterazioni medio: {np.mean(iters)}")

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

run_perceptron(X1, Y1)  # setosa vs virginica: separabili
run_perceptron(X2, Y2)  # setosa vs versicolor: separabili
run_perceptron(X3, Y3)  # versicolor vs virginica: NON separabili