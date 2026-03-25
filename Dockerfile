FROM python:3.12-slim

WORKDIR /app

# System deps for spaCy, PyMuPDF, audio processing, and general build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA build)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
