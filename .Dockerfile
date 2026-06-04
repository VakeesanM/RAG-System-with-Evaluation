FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN python -m pip install -r requirement.txt

COPY . .

ENV OPENAI_API_KEY=OPENAI_API_KEY

EXPOSE 8501

CMD ["streamlit" "run" "RAGSystem/app.py"]