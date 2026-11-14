# Utiliser une image officielle Python légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour PyTorch / Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les exigences Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# S'assurer que python-multipart est bien inclus
RUN pip install --no-cache-dir python-multipart

# Copier le code de l'application et le modèle entraîné
COPY main.py .
COPY model_rice_disease.pth .

# Exposer le port (purement indicatif pour Docker)
EXPOSE 8000

# Commande dynamique pour Render et le développement local
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}