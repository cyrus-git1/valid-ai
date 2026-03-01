FROM python:3.12-slim

WORKDIR /app

# System deps for spaCy, PyMuPDF, and general build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.main"]
