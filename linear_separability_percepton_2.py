import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generiamo 100 campioni, divisi in 2 classi (centers=2), con 2 feature (X1, X2)
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.5, random_state=42)

print("Formato delle feature (X):", X.shape)
print("Prime 5 feature (X):", X[:5]) # Saranno [[X1, X2], [X3, X4], ... , [Xn-1, Xn]]
print("Prime 5 etichette (Y):", Y[:5]) # Saranno 0 o 1

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import Perceptron
# max_iter è il numero massimo di epoche (passaggi sull'intero dataset).
# tol è il criterio di arresto (se l'errore non migliora di questa quantità, si ferma).
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Addestriamo il modello
model.fit(X_train, Y_train)

# Il modello ha trovato i pesi (w) e il bias (b) per l'equazione: w1*x1 + w2*x2 + b = 0
print(f"Pesi trovati (w1, w2): {model.coef_[0]}")
print(f"Bias trovato (b): {model.intercept_[0]}")

from sklearn.metrics import accuracy_score

# Facciamo le previsioni sul test set
Y_pred = model.predict(X_test)

# Confrontiamo le etichette reali con quelle previste
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuratezza del Perceptron: {accuracy * 100:.2f}%")


# Creiamo il grafico a dispersione dei dati di test
plt.scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], color='blue', label='Classe 0')
plt.scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], color='red', label='Classe 1')

# Estraiamo i pesi per calcolare la retta
w1, w2 = model.coef_[0]
b = model.intercept_[0]

# Creiamo due punti estremi sull'asse X per tracciare la linea
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# L'equazione del confine è: w1*x1 + w2*x2 + b = 0
# Esplicitiamo x2 per disegnarla: x2 = -(w1*x1 + b) / w2
x2_min = -(w1 * x1_min + b) / w2
x2_max = -(w1 * x1_max + b) / w2

plt.plot([x1_min, x1_max], [x2_min, x2_max], color='black', linestyle='--', label='Confine di decisione')

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.legend()
plt.title("Perceptron: Separazione in Semispazi")
plt.show()