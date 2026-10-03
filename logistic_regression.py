import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generiamo 200 campioni con 2 feature (X1, X2)
X, Y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,  # n_informative=2 significa che entrambe le feature sono utili per la classificazione.
    random_state=42,
    n_clusters_per_class=1,
    class_sep=1.0,  # class_sep=1.0 regola quanto le classi sono distanti (valori più bassi = più sovrapposizione).
)

print("Dimensioni del dataset (feature):", X.shape)
print("Dimensioni del dataset (label):", Y.shape[0])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)
print(f"Istanze di Training: {len(X_train)}")
print(f"Istanze di Test: {len(X_test)}")


from sklearn.linear_model import LogisticRegression

# Inizializziamo il modello.
model = LogisticRegression(
    solver="lbfgs", random_state=42
)  # solver è l'algoritmo matematico per ottimizzare i pesi.

# Avviamo l'apprendimento
model.fit(X_train, Y_train)

print(f"Pesi appresi (w1, w2): {model.coef_[0]}")
print(f"Bias appreso (b): {model.intercept_[0]}")


# Previsione standard (Classe 0 o 1)
Y_pred = model.predict(X_test)

# Previsione probabilistica (Probabilità di essere Classe 0, Probabilità di essere Classe 1)
Y_prob = model.predict_proba(X_test)

print("\nAnalisi delle prime 3 istanze di test:")
for i in range(3):
    print(f"Istanza {i}: Classe predetta = {Y_pred[i]}")
    print(f"  -> Probabilità Classe 0: {Y_prob[i][0]*100:.1f}%")
    print(f"  -> Probabilità Classe 1: {Y_prob[i][1]*100:.1f}%\n")


from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 1. Accuratezza globale
acc = accuracy_score(Y_test, Y_pred)
print(f"Accuratezza del modello: {acc * 100:.2f}%")

# 2. Matrice di confusione
cm = confusion_matrix(Y_test, Y_pred)

# Estraiamo i 4 valori usando ravel() che "appiattisce" la matrice 2x2 in un array 1D
tn, fp, fn, tp = cm.ravel()

# Creiamo le etichette unendo le sigle ai numeri (es. "TN\n24")
labels = [f"TN\n{tn}", f"FP\n{fp}", f"FN\n{fn}", f"TP\n{tp}"]

# Rimodelliamo la lista in una matrice 2x2 per farla combaciare con la heatmap
labels = np.asarray(labels).reshape(2, 2)

# Disegniamo la matrice
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=labels,
    fmt="", # per formattare le stringhe non come numeri
    cmap="Blues",
    xticklabels=["Previsto 0", "Previsto 1"],
    yticklabels=["Reale 0", "Reale 1"],
)

plt.title("Matrice di Confusione con TN, FP, FN, TP")
plt.show()
