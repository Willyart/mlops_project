from google.cloud import storage
import requests
import urllib.parse
import os
from roboflow import Roboflow
import glob

# ************* SET THESE VARIABLES *************
GCS_BUCKET_NAME = "mlopps-roboflow-ds"
ROBOFLOW_API_KEY = "4m9GkHwOJE416dHxx26p"
ROBOFLOW_PROJECT_NAME = "mlops-aplvu"
GOOGLE_APPLICATION_CREDENTIALS = "mlopps-bucket-service-account-file.json"
# ***********************************************

def get_gcs_signed_url(bucket_name: str, blob_name: str) -> str:
    """Generate a signed URL for a GCS object."""
    storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    
    url = blob.generate_signed_url(
        version="v4",
        expiration=3600,  # 1 hour in seconds
        method="GET"
    )
    return url

def get_gcs_objects(bucket_name: str) -> list:
    """Fetch the list of object keys in the given GCS bucket."""
    storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    object_names = []
    for blob in blobs:
        object_names.append(blob.name)
    return object_names





def upload_to_roboflow_annotations(api_key: str, project_name: str, image_name: str, annotation_path: str, image_id: str):
    """
    Récupère les annotations YOLO depuis une URL (fichier texte en GCS) 
    et les upload en format JSON à Roboflow.
    """
    API_URL = "https://api.roboflow.com"

    # Récupérer le contenu du fichier d'annotations depuis l'URL GCS signée
    # ann_response = requests.get(blob_url)
    # if ann_response.status_code != 200:
    #     print(f"Impossible de récupérer les annotations depuis {blob_url}")
    #     return False

    annotation_data = ann_response.text.strip()

    # Convertir les annotations YOLO en JSON
    # YOLO format (par ligne) : class_id x_center y_center width height (valeurs normalisées)
   
    annotation_data = annotation_data.replace("\\\\", "")  # Remplacer les doubles antislashs par rien
    annotation_data = annotation_data.replace('"', '\\"')  # Échapper les guillemets doubles

    body = {
        "annotationFile": annotation_data,
        "labelmap":  {"0":"Solar panel", "1":"passage_pieton"}
    }

    print(body)

    # Construire l’URL Roboflow : on utilise le nom de l'image sans l'extension
    base_name = os.path.splitext(blob_name)[0]
    # upload_url = f"{API_URL}/dataset/{project_name}/annotate/{base_name}"

    upload_url = "".join([
        API_URL + "/dataset/" + project_name + "/annotate/" + image_id,
        "?api_key=" + api_key,
        "&name=" + blob_name,
    ])
    print(upload_url)
    # Envoyer la requête POST avec les annotations en JSON
    response = requests.post(upload_url, json=body)

    if response.status_code == 200:
        print(f"Annotations pour {blob_name} uploadées avec succès dans {project_name}")
        return True
    else:
        print(f"Echec de l'upload de {blob_name}. Erreur: {response.content.decode('utf-8')}")
        return False

def upload_to_roboflow(api_key: str, project_name: str, image_path: str, img_name='', split="train"):
    """Upload an image to Roboflow."""
    API_URL = "https://api.roboflow.com"
    
    # with open(image_path, 'rb') as image_file:
    #     # Préparer les données pour l'upload (inclut l'image en tant que fichier)
    #     files = {
    #         'image': (img_name if img_name else image_path.split("/")[-1], image_file, 'image/jpeg')  # Ajuste le type MIME si nécessaire
    #     }
        
    #     # Paramètres supplémentaires
    #     data = {
    #         'api_key': api_key,
    #         'name': img_name,
    #         'split': split
    #     }

    #     # Construire l'URL de l'API d'upload
    #     upload_url = f"{API_URL}/dataset/{project_name}/upload"

    #     # Envoyer la requête POST avec le fichier et les paramètres
    #     response = requests.post(upload_url, data=data, files=files)
    with open(image_path, 'rb') as image_file:
        files = {
            'image': (img_name if img_name else image_path.split("/")[-1], image_file, 'image/jpeg')  # Ajuste le type MIME si nécessaire
        }
        upload_url = "".join([
            API_URL + "/dataset/" + project_name + "/upload",
            "?api_key=" + api_key,
            "&name=" + img_name,
            "&split=" + split,
            "&image=" + files,
        ])
        response = requests.post(upload_url)

    # Check response code
    if response.status_code == 200:
        print(f"Successfully uploaded {img_name} to {project_name}")
        return response
    else:
        print(f"Failed to upload {img_name}. Error: {response.content.decode('utf-8')}")
        return False
    


if __name__ == "__main__":
    # Fetch list of available blobs
    # available_blobs = get_gcs_objects(GCS_BUCKET_NAME)
    
    # print(available_blobs)
    # Optional: Filter the blobs here
    # e.g., available_blobs = [blob for blob in available_blobs if "some_condition"]
    
    response = ""

    # Listes des chemins de dossiers
    dirs = [
        "../datasets/data/test/labels",

        "../datasets/data/test/images",
        "../datasets/data/to_predict",
        "../datasets/data/train/images",
        "../datasets/data/train/labels"
    ]


    rf = Roboflow(api_key="4m9GkHwOJE416dHxx26p")
    project = rf.workspace("alex-07bbm").project("mlops-aplvu")

    # Dossier contenant les images et annotations
    image_dir = "../datasets/data/train/images/"
    label_dir = "../datasets/data/train/labels/"
    file_extension_type = ".JPG"

    # Obtenez toutes les images avec l'extension spécifiée
    image_glob = glob.glob(image_dir + '*' + file_extension_type)

    # Pour chaque image, trouver et uploader l'image et son annotation associée
    for image_path in image_glob:
        # Obtenir le nom de l'image sans l'extension
        image_name = os.path.basename(image_path).replace(file_extension_type, '')
        
        # Trouver le fichier d'annotation correspondant (même nom mais extension .txt)
        annotation_filename = os.path.join(label_dir, image_name + '.txt')
        
        if os.path.exists(annotation_filename):  # Vérifier si le fichier d'annotation existe
            print(f"Uploading {image_name} and its annotation...")
            response = project.single_upload(
                image_path=image_path,
                annotation_path=annotation_filename,
                # Optionnels:
                # split='train',  # Si tu veux spécifier le split
                # tag_names=['tag1', 'tag2'],  # Si tu veux ajouter des tags
                # is_prediction=False,  # Si ce n'est pas une image de prédiction
            )
            print(response)  # Afficher la réponse de l'upload
        else:
            print(f"Annotation file for {image_name} not found. Skipping upload.")
    

