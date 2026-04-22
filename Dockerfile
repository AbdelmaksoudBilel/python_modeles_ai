FROM python:3.10-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur obligatoire pour HF
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Mise à jour de pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Étape critique : Installation isolée des frameworks lourds
RUN pip install --no-cache-dir torch==2.3.0 tensorflow==2.16.1

# Installation du reste des dépendances
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY --chown=user . .

# Exposition du port requis par Hugging Face
EXPOSE 7860

# Commande de lancement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]