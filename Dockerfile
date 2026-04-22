FROM python:3.10-slim

# Mise à jour des paquets et installation des dépendances système corrigées
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ... le reste du Dockerfile demeure identique ...
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Étape critique : Installation isolée des frameworks ET du patch de compatibilité Keras
RUN pip install --no-cache-dir torch==2.3.0 tensorflow==2.16.1 tf-keras

# Installation du reste
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]