import numpy as np

# Fissiamo il "seed" per la generazione dei numeri assicurandoci la riproducibilità
np.random.seed(42)

# Creiamo un dataset fittizio: X = metri quadri di una casa, Y = prezzo
# X rappresenta le nostre feature 
X = 2 * np.random.rand(100, 1) # Creiamo 100 valori casuali tra 0 e 2

# Fissiamo la funzione di etichettatura reale f: X->Y, con f(X)=Y : Y=(4+3)*X. 
Y = (4 + 3) * X + np.random.randn(100, 1) # Aggiungiamo np.random.randn per creare disturbo.

from sklearn.model_selection import train_test_split

# Dividiamo i dati: 80% per l'addestramento (train) e 20% per il test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Dati di addestramento: {X_train.shape[0]} campioni")
print(f"Dati di test: {X_test.shape[0]} campioni")

from sklearn.linear_model import LinearRegression

# 1. Inizializziamo il modello
model = LinearRegression()

# 2. Addestriamo il modello sui dati di train
model.fit(X_train, Y_train)

# Vediamo cosa ha imparato (dovrebbe avvicinarsi a intercetta=4 e coefficiente=3)
print(f"Intercetta trovata (punto di partenza): {model.intercept_[0]:.2f}")
print(f"Coefficiente trovato (pendenza): {model.coef_[0][0]:.2f}")

from sklearn.metrics import mean_squared_error

# Chiediamo al modello di prevedere le Y a partire dalle X di test
Y_pred = model.predict(X_test)

# Calcoliamo l'Errore Quadratico Medio (MSE): # Misura la distanza media al quadrato tra le previsioni e i valori reali
mse = mean_squared_error(Y_test, Y_pred)
print(f"Errore Quadratico Medio (MSE): {mse:.2f}")

import matplotlib.pyplot as plt

# Disegniamo i punti reali del test set (blu)
plt.scatter(X_test, Y_test, color='blue', label='Dati reali (Test)')

# Disegniamo la retta delle previsioni del modello (rossa)
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Previsione del modello')

# Aggiungiamo etichette per rendere il grafico leggibile
plt.xlabel("Metri Quadri (X)")
plt.ylabel("Prezzo (y)")
plt.legend()
plt.title("La retta trovata dalla Regressione Lineare")

# Mostriamo il grafico
plt.show()