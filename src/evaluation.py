#!/usr/bin/env python3

import argparse
import os
from os import path
from ultralytics import YOLO
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle YOLO sur un ensemble de validation.")
    parser.add_argument('--base_dir', type=str, default='./datasets/data',
                        help='Chemin de base vers le dataset contenant train/ et test/.')
    parser.add_argument('--model_path', type=str, default='./models/crossingsSolars_yolo11/weights/best.pt',
                        help='Chemin vers le poids du modèle YOLO entraîné (ex: best.pt).')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Nom du répertoire de sortie pour les résultats d\'évaluation (sera créé dans base_dir).')

    args = parser.parse_args()

    val_dir = path.join(args.base_dir, 'test', 'images')
    output_dir = path.join(args.base_dir, args.output_dir)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    print(f"Chargement du modèle depuis : {args.model_path}")
    model = YOLO(args.model_path)
    data_yaml = path.abspath("datasets/data/data.yaml")
    print(f"Évaluation sur les images du dossier : {val_dir}")

    # Effectuer l'évaluation
    results = model.val(
        data=data_yaml,  # Utilisation du fichier data.yaml
        imgsz=640,  # Taille des images
        batch=8,    # Taille du batch
        split='val' # Utilisation du jeu de validation
    )

    # Extraire les métriques (mAP@50 comme accuracy simulée, faute de `accuracy` explicite)
    val_accuracy = float(results.box.map)  # mAP@50:95
    val_loss = float(results.box.mr)  # Rappel moyen utilisé comme proxy pour "loss" faute de mieux

    # Affichage des métriques
    print(f"Validation loss (approximation): {val_loss:.2f}")
    print(f"Validation accuracy: {val_accuracy:.2f}")
    

    # Sauvegarder les métriques dans un fichier JSON
    metrics = {"val_loss": val_loss, "val_accuracy": val_accuracy}
    metrics_file = path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées dans : {metrics_file}")