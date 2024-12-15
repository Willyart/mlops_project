#!/usr/bin/env python3

import argparse
import os
from os import path
from ultralytics import YOLO
import json
import shutil

def get_latest_folder(directory):
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not folders:
        return None  

    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder

def copy_folder(src_folder, dest_directory):
    folder_name = os.path.basename(src_folder)
    dest_path = os.path.join(dest_directory, folder_name)

    shutil.copytree(src_folder, dest_path)
    return dest_path

def rename_folder(folder_path, new_name):
    parent_dir = os.path.dirname(folder_path)
    new_path = os.path.join(parent_dir, new_name)
    
    if os.path.exists(new_path):
        print(f"Le dossier cible {new_path} existe déjà, mise à jour en cours...")
        shutil.rmtree(new_path)
    
    shutil.move(folder_path, new_path)
    print(f"Dossier renommé avec succès : {new_path}")
    return new_path

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

    os.makedirs(output_dir, exist_ok=True)

    print(f"Chargement du modèle depuis : {args.model_path}")
    model = YOLO(args.model_path)
    data_yaml = path.abspath("datasets/data/data.yaml")
    print(f"Évaluation sur les images du dossier : {val_dir}")

    results = model.val(
        data=data_yaml,  
        imgsz=640,  
        batch=8,   
        split='val' 
    )

    val_accuracy = float(results.box.map)
    val_loss = float(results.box.mr) 

    folder_last_run = get_latest_folder(path.abspath("runs/detect/"))

    to_evaluation_folder = copy_folder(folder_last_run, path.abspath("datasets/data/evaluation/"))

    rename_folder(to_evaluation_folder, "plots")

    print(f"Validation loss (approximation): {val_loss:.2f}")
    print(f"Validation accuracy: {val_accuracy:.2f}")
    

    metrics = {"val_loss": val_loss, "val_accuracy": val_accuracy}
    metrics_file = path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métriques sauvegardées dans : {metrics_file}")