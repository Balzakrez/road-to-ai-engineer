from sqlalchemy import Column, String, Float, Date,Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

# Creazione classe base
Base = declarative_base()

# Creazione engine
engine = create_engine("sqlite:///finance.db", connect_args={"check_same_thread": False})


class TransazioneModel(Base):
    __tablename__ = "transazioni"
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_contabile = Column(Date)
    data_valuta = Column(Date)
    importo = Column(Float)
    descrizione = Column(String)
    tipo_movimento = Column(String)
    categoria = Column(String)

