# Utiliser une image de base avec Python
FROM python:3.8-slim

# Installer les dépendances
RUN pip install --no-cache-dir \
    ultralytics \
    pyyaml \
    torch \
    opencv-python


# Définir le répertoire de travail
WORKDIR /app


RUN git clone https://github.com/Willyart/mlops_project.git /app

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN dvc pull  

EXPOSE 5000

# CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "5000"]
