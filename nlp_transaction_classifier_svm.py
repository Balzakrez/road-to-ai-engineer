import pandas as pd
import random

# 1. GENERAZIONE DEI DATI FITTIZI
# Creiamo delle descrizioni tipiche da estratto conto e la loro categoria target (12)
dati_base = [
    ("PAGAMENTO POS SUPERMERCATO CONAD", "Alimentari"),
    ("ADDEBITO DIRETTO NETFLIX", "Intrattenimento"),
    ("BONIFICO AFFITTO MARIO ROSSI", "Casa"),
    ("PAGAMENTO POS RISTORANTE DA PINO", "Ristorazione"),
    ("ADDEBITO BOLLETTA ENEL ENERGIA", "Utenze"),
    ("SPESA ESSELUNGA", "Alimentari"),
    ("RINNOVO ABBONAMENTO SPOTIFY", "Intrattenimento"),
    ("PAGAMENTO PIZZERIA", "Ristorazione"),
    ("ADDEBITO TRENITALIA", "Trasporti"),
    ("PAGAMENTO POS BENZINA ENI", "Trasporti"),
    ("FARMACIA COMUNALE", "Salute"),
    ("ADDEBITO TIM SPA", "Utenze")
]

# Moltiplichiamo i dati e aggiungiamo un po' di casualità per simulare un dataset più corposo (120 righe)
dataset_esteso = dati_base * 10 
random.seed(42)
random.shuffle(dataset_esteso)

# 2. CREAZIONE DEL DATAFRAME CON PANDAS
# Inizializziamo il DataFrame specificando i nomi delle due colonne
df = pd.DataFrame(dataset_esteso, columns=["Descrizione", "Categoria"])

# 3. SALVATAGGIO IN CSV
nome_file = "dataset_transazioni.csv"
df.to_csv(nome_file, index=False) # Salviamo il file su disco senza esportare l'indice numerico delle righe
print(f"File '{nome_file}' creato con successo!\n")


# 4. LETTURA E ISPEZIONE DATI (Quello che farai sempre all'inizio di un progetto ML)
# Carichiamo il dataset dal file appena creato
df_caricato = pd.read_csv(nome_file)

# .head() mostra le prime 5 righe del DataFrame
print("Le prime 5 righe del dataset:")
print(df_caricato.head())

# .value_counts() conta quante transazioni ci sono per ogni categoria
print("\nDistribuzione delle categorie:")
print(df_caricato["Categoria"].value_counts())


from sklearn.feature_extraction.text import TfidfVectorizer

# Separiamo le feature (X, il testo in input) dal target (Y, l'etichetta da prevedere)
X_testo = df_caricato["Descrizione"]
Y = df_caricato["Categoria"]

# 5. INIZIALIZZAZIONE DEL VETTORIZZATORE
# max_features limita il vocabolario alle N parole più rilevanti (utile quando hai milioni di righe)
vectorizer = TfidfVectorizer(max_features=100)

# 6. TRASFORMAZIONE (FIT E TRANSFORM)
# Il metodo fit_transform() esegue due passaggi fondamentali: 
# - fit: legge tutto il dataset e costruisce il dizionario delle parole conosciute
# - transform: converte ogni frase in un vettore numerico
X_numerico = vectorizer.fit_transform(X_testo)

# 7. ISPEZIONE DEL RISULTATO
vocabolario = vectorizer.get_feature_names_out()
print("Vocabolario appreso (prime 10 parole):")
print(vocabolario[:10])

print(f"\nFormato originale dei dati (X_testo): {X_testo.shape}")
print(f"Formato vettorializzato (X_numerico): {X_numerico.shape}")

# Visualizziamo come il modello vede la prima transazione
print(f"\nTesto originale: '{X_testo[0]}'")
print("Rappresentazione matematica (Sparse Matrix):")
print(X_numerico[0])

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# 8. SPLIT DEI DATI (sui vettori numerici, non sul testo grezzo)
# X_numerico e Y arrivano dallo step TF-IDF precedente
X_train, X_test, Y_train, Y_test = train_test_split(X_numerico, Y, test_size=0.2, random_state=42)


# 9. INIZIALIZZAZIONE E ADDESTRAMENTO DELLA SVM
# LinearSVC è una versione ottimizzata delle SVM per i kernel lineari (ideale e velocissima per il testo)
model = LinearSVC(random_state=42)
model.fit(X_train, Y_train)

# 10. VALUTAZIONE SUL TEST SET
Y_pred = model.predict(X_test)
print("Report di Classificazione sulle categorie:")
# classification_report mostra le metriche avanzate (precision, recall, f1-score) per ogni singola classe
print(classification_report(Y_test, Y_pred))

# 11. IL TEST DEFINITIVO: UNA NUOVA TRANSAZIONE
# Simuliamo l'arrivo di un nuovo movimento dal conto corrente
nuova_transazione = ["PAGAMENTO POS RISTORANTE DA MICHELE"]

# ATTENZIONE: Usiamo solo transform(), non fit_transform(), perché il vocabolario è già fissato e appreso
nuova_transazione_vettorizzata = vectorizer.transform(nuova_transazione)
previsione = model.predict(nuova_transazione_vettorizzata)

print(f"\nLa transazione '{nuova_transazione[0]}' è stata classificata come: {previsione[0]}")