import json
import os
import random
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import delete, desc, select, func
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from database import TransazioneModel, Base, engine

# Controlla se il db è già stato creato
DB_FILE = "finance.db"
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

# Creazione app
app = FastAPI()


# Monta la cartella 'static' per servire file CSS/JS se necessari in futuro
app.mount("/static", StaticFiles(directory="static"), name="static")

# Creazione db
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Crea fisicamente le tabelle nel database SQLite se non esistono già
Base.metadata.create_all(bind=engine)


def addestra_e_classifica(df_clean_duplicates: pd.DataFrame):
    
    # 1. Dataset di training iniziale (le frasi e le relative categorie corrette)
    training_file = "training_data.json"
    if os.path.exists(training_file):
        with open(training_file, "r", encoding="utf-8") as f:
            dati_base = json.load(f)
    else:
        dati_base = [("SPESA GENERERICA", "Generica")]

    dati_base = dati_base * 10
    random.shuffle(dati_base)

    # Separiamo testi e categorie del dataset di training
    X_train_text = [tlp[0] for tlp in dati_base]
    Y_train = [tlp[1] for tlp in dati_base]

    # Dividiamo il dataset di base (es. 75% training, 25% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_text, Y_train, test_size=0.25, random_state=42)

    # 2. Inizializzazione e fit del vectorizer e del modello SVM
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_vector = vectorizer.fit_transform(X_train)

    model = LinearSVC(random_state=42)
    model.fit(X_train_vector, Y_train)

    # Valutiamo sul test
    X_test_vector = vectorizer.transform(X_test)
    predizioni = model.predict(X_test_vector)

    accuratezza = accuracy_score(Y_test, predizioni)
    print(f"Accuratezza del modello: {accuratezza * 100:.2f}%")

    # 3. Classificazione delle transazioni reali in arrivo dal file Excel
    transazioni_da_salvare = []
    for idx, row in  df_clean_duplicates.iterrows():
        descrizione = row["Descrizione operazioni"]

        # 3.1 Vettorizziamo la nuova descrizione usando il vocabolario già appreso (transform, non fit_transform)
        descrizione_vettorizzata = vectorizer.transform([descrizione])

        # 3.2 Prediciamo la categoria
        categoria_predetta = model.predict(descrizione_vettorizzata)[0]

        transazione = TransazioneModel(
            data_contabile=row["Data Contabile"],
            data_valuta=row["Data Valuta"],
            importo=row["Importo (euro)"],
            descrizione=descrizione,
            tipo_movimento=row["Tipo Movimento"],
            categoria=categoria_predetta # Assegniamo la categoria predetta dalla SVM!
        )
        transazioni_da_salvare.append(transazione)

    return transazioni_da_salvare


@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.get("/api/hello")
def saluta():
    return {"messaggio": "Hello World"}

@app.get("/api/analytics/riepilogo")
def riepilogo(db: Session = Depends(get_db)):
    totale_entrate = db.scalar(
        select(func.sum(TransazioneModel.importo))
        .where(TransazioneModel.tipo_movimento=="ENTRATA")
    ) or 0.0
    
    totale_uscite = db.scalar( 
        select(func.sum(TransazioneModel.importo))
        .where(TransazioneModel.tipo_movimento=="USCITA")
    ) or 0.0
    
    saldo_netto = totale_entrate - totale_uscite

    return {
        "totale_entrate": round(totale_entrate, 2),
        "totale_uscite": round(totale_uscite, 2),
        "saldo_netto": round(saldo_netto, 2)
    }


@app.get("/api/transizioni")
def lista_transizioni(db: Session = Depends(get_db)):
    transazioni = db.scalars(select(TransazioneModel)).all()
    return [
        {
            "id": t.id,
            "data": t.data_contabile,
            "descrizione": t.descrizione,
            "importo": t.importo,
            "tipo": t.tipo_movimento,
            "categoria": t.categoria
        }
        for t in transazioni
    ]

@app.get("/api/analytics/reset")
def reset(db: Session = Depends(get_db)):
    statement = delete(TransazioneModel)
    db.execute(statement)
    db.commit()
    return {"messaggio": "Database resettato con successo"}


@app.post("/api/transizioni/import")
def importa(file_xlsx: UploadFile = File(...), db: Session = Depends(get_db)): 
    df = pd.read_excel(file_xlsx.file, header=2)

    print(df.head())
    print(df.columns.tolist())
    print(df.shape)

    df_clean = df.dropna(subset=["Descrizione operazioni"]).copy()

    df_clean["Tipo Movimento"] = df_clean["Importo (euro)"].apply(lambda x: "USCITA" if x < 0 else "ENTRATA")
    df_clean["Importo (euro)"] = df_clean["Importo (euro)"].abs()

    df_clean_duplicates = df_clean.drop_duplicates(subset=["Data Contabile", "Importo (euro)", "Descrizione operazioni"])

    transazioni_da_salvare = addestra_e_classifica(df_clean_duplicates)
    
    db.add_all(transazioni_da_salvare)
    db.commit()
    

