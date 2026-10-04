# Poste Finance Analyzer

Backend API per l'ingestione, l'elaborazione e la categorizzazione automatica tramite Machine Learning dei movimenti bancari.

## Struttura del Progetto

```text
poste-finance-analyzer/
├── app.py                 # Core API, pipeline ETL e integrazione ML
├── database.py            # Configurazione e modelli ORM (SQLAlchemy)
├── training_data.json     # Dataset di addestramento SVM (escluso dal repository)
├── ListaMovimenti.xlsx    # File di input di esempio (Excel)
└── finance.db             # Database SQLite locale (generato all'avvio)

```

## Installazione

Attivare l'ambiente virtuale e installare le dipendenze necessarie:

```bash
pip install fastapi uvicorn pandas sqlalchemy scikit-learn openpyxl

```

## Avvio del Server

Avviare l'applicazione tramite Uvicorn:

```bash
uvicorn app:app --reload

```

La documentazione interattiva dell'API (Swagger UI) sarà disponibile all'indirizzo:

`[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

## Architettura di Classificazione (Machine Learning)

Il sistema integra una pipeline NLP (Natural Language Processing) basata su Scikit-Learn per superare i limiti delle classiche regole basate su parole chiave o espressioni regolari.

L'elaborazione si articola in due step analitici durante la fase di importazione:

1. **Vettorizzazione (TF-IDF):** Le descrizioni testuali delle transazioni vengono elaborate da un `TfidfVectorizer`. Questo componente converte il testo in vettori numerici, assegnando un peso maggiore ai termini distintivi (es. "AMAZON", "ENEL", "SUMUP") e penalizzando le parole generiche presenti in tutte le causali.
2. **Classificazione (Support Vector Machine):** Un modello `LinearSVC` viene addestrato dinamicamente sui vettori generati. La SVM proietta i dati in uno spazio multidimensionale e calcola i margini ottimali per separare le diverse categorie finanziarie.

Durante l'upload del file Excel, le nuove descrizioni subiscono la medesima trasformazione TF-IDF e vengono sottoposte al modello per la predizione. La categoria risultante viene infine associata al record e persistita nel database SQLite.

## Configurazione Dataset (`training_data.json`)

Per addestrare il modello senza esporre dati finanziari sensibili nel codice sorgente, i pattern di categorizzazione vengono caricati dinamicamente da un file esterno. La precisione predittiva del modello SVC è direttamente proporzionale alla varietà degli esempi inseriti in questo dataset.

Creare il file `training_data.json` nella root del progetto seguendo questa struttura (array di coppie `["Descrizione o pattern", "Categoria"]`):

```json
[
    ["PAGAMENTO POS SUPERMERCATO", "Alimentari"],
    ["Pagamento E-Commerce AMAZON", "Shopping"],
    ["ADDEBITO ABBONAMENTO", "Intrattenimento"],
    ["Pagamento Apple Pay RISTORANTE", "Ristorazione"],
    ["PAGAMENTO POS BENZINA", "Trasporti"],
    ["ADDEBITO BOLLETTA ENERGIA", "Utenze"],
    ["PRELIEVO ATM", "Prelievi"],
    ["BONIFICO STIPENDIO", "Entrate"],
    ["PAGAMENTO FARMACIA", "Salute"],
    ["CANONE CARTA", "Banca"]
]

```

**Nota sulla sicurezza:**

Assicurarsi che il file `training_data.json` sia incluso nel file `.gitignore` per prevenire il tracciamento e la pubblicazione di informazioni personali.