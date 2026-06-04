FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN python -m pip install -r requirement.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "RAGSystem/app.py", "--server.address=0.0.0.0", "--server.port=8501"]