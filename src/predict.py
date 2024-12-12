import argparse
import os
import cv2
import glob
import random
from os import path
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire avec un modèle YOLO sur un ensemble d'images.")
    parser.add_argument('--base_dir', type=str, default='./data',
                        help='Chemin de base vers le dataset contenant to_predict/.')
    parser.add_argument('--model_path', type=str, default='./models/crossingsSolars_yolo11/weights/best.pt',
                        help='Chemin vers le poids du modèle YOLO entraîné (ex: best.pt).')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Nom du répertoire de sortie pour les images prédictes (sera créé dans base_dir).')

    args = parser.parse_args()

    test_images_path = path.join(args.base_dir, 'to_predict')
    output_dir = path.join(args.base_dir, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(args.model_path)
    image_paths = glob.glob(os.path.join(test_images_path, '*'))
    random.shuffle(image_paths)

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        img = cv2.imread(img_path)
        results = model(img)
        result_img = results[0].plot()

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        print("Output path : " + output_path)
        cv2.imwrite(output_path, result_img)
