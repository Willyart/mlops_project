import argparse
from os import path
from ultralytics import YOLO
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un modèle YOLO sur un dataset.")
    parser.add_argument('--base_dir', type=str, default='./data',
                        help='Chemin de base vers le dataset contenant train/ et test/.')
    parser.add_argument('--model_path', type=str, default='./models/yolo11s.pt',
                        help='Chemin vers le modèle YOLO pré-entraîné.')
    parser.add_argument('--project_dir', type=str, default='.',
                        help='Répertoire où seront stockés les résultats (chemin relatif ou absolu).')
    parser.add_argument('--name', type=str, default='./models/crossingsSolars_yolo11',
                        help='Nom du sous-dossier créé dans project_dir pour cette session.')
    
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques d\'entraînement.')
    parser.add_argument('--imgsz', type=int, default=640, help='Taille des images d\'entrée.')
    parser.add_argument('--batch', type=int, default=8, help='Taille du batch.')

    args = parser.parse_args()

    train_params = yaml.safe_load(open("params.yaml"))["train"]

    data_yaml = path.abspath("datasets/data/data.yaml")
    model = YOLO(args.model_path)
    
    model.train(
        data=data_yaml,
        epochs=train_params["epochs"],
        imgsz=args.imgsz,
        batch=train_params["batch"],
        project=args.project_dir,
        name=args.name
    )
