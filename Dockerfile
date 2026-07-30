FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY src/ /app/src/

WORKDIR /app/src

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501","--server.enableXsrfProtection=false", "--server.enableCORS=false"]