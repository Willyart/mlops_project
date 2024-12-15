import os
import json
from os import path
import argparse

def class_name_to_index(class_name):
    
    mapping = {
        'panneaux_solaires': 0,
        'passage_pieton': 1
    }
    return mapping.get(class_name, -1)

def convert_labelme_json_to_yolo_format(json_path, labels_dir):

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erreur lors de la lecture du fichier {json_path}: {e}")
        return

    image_width = data.get('imageWidth')
    image_height = data.get('imageHeight')

    if not image_width or not image_height:
        print(f"Dimensions de l'image manquantes dans {json_path}, conversion ignorée.")
        return

    output_data = []

    for shape in data.get('shapes', []):
        class_name = shape['label']
        class_index = class_name_to_index(class_name)

        if class_index == -1:
            print(f"Classe inconnue '{class_name}' dans {json_path}, ignorée.")
            continue

        points = shape['points']
        x_min = min([p[0] for p in points])
        x_max = max([p[0] for p in points])
        y_min = min([p[1] for p in points])
        y_max = max([p[1] for p in points])

        x_center = ((x_min + x_max) / 2.0) / image_width
        y_center = ((y_min + y_max) / 2.0) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        output_data.append(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    base_name = path.splitext(path.basename(json_path))[0]
    output_file = path.join(labels_dir, base_name + '.txt')

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_data))

    print(f"Fichier YOLO créé : {output_file}")



def process_dataset(dataset_dir):

    print(f"Traitement du répertoire : {dataset_dir}")
    labels_dir = path.join(path.dirname(dataset_dir), 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.json'):
            json_path = path.join(dataset_dir, file_name)
            print(f"Conversion : {json_path}")
            convert_labelme_json_to_yolo_format(json_path, labels_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir les annotations Labelme en YOLO et générer data.yaml.")

    parser.add_argument('--base_dir', type=str, default='./datasets/data', help='Chemin de base vers le dataset contenant train/ et test/.')
    

    args = parser.parse_args()

    base_dir = args.base_dir
    train_dir = path.join(base_dir, 'train', 'images')
    test_dir = path.join(base_dir, 'test', 'images')

    process_dataset(train_dir)
    process_dataset(test_dir)

    data_yaml_content = """train: train/images
val: test/images
nc: 2
names: ['panneaux_solaires', 'passage_pieton']
    """
    data_yaml_path = path.join(base_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    print(f"Fichier data.yaml créé : {data_yaml_path}")
