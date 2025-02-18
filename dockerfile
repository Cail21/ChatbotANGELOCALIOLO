# Usa l'immagine ufficiale di Python 3.12
FROM python:3.12-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia i file del progetto nel container
COPY . /app

# Installa le dipendenze
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --upgrade langchain  # <--- Aggiunto aggiornamento di LangChain

# Espone la porta usata da Streamlit
EXPOSE 8501

# Comando per avviare Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
