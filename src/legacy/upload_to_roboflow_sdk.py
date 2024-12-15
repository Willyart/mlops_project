from google.cloud import storage
import requests
import urllib.parse
import os

# ************* SET THESE VARIABLES *************
GCS_BUCKET_NAME = "bucket-mlops-project-a-b-d-g-m"
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


def upload_to_roboflow_annotations(api_key: str, project_name: str, presigned_url: str, img_name='', split="train"):
    """Upload an image to Roboflow."""
    API_URL = "https://api.roboflow.com"
    if img_name == '':
        img_name = presigned_url.split("/")[-1]
        img_name = img_name.split("?")[0]
    

    upload_url = "".join([
        API_URL + "/dataset/" + project_name + "/annotate/" + img_name.split(".")[0],
        "?api_key=" + api_key,
        "&name=" + img_name,
    ])
    response = requests.post(upload_url)

    # Check response code
    if response.status_code == 200:
        print(f"Successfully uploaded {img_name} to {project_name}")
        return response
    else:
        print(f"Failed to upload {img_name}. Error: {response.content.decode('utf-8')}")
        return False


def upload_to_roboflow_annotations_2(api_key: str, project_name: str, blob_name: str, blob_url: str, image_id: str):
    """
    Récupère les annotations YOLO depuis une URL (fichier texte en GCS) 
    et les upload en format JSON à Roboflow.
    """
    API_URL = "https://api.roboflow.com"

    # Récupérer le contenu du fichier d'annotations depuis l'URL GCS signée
    ann_response = requests.get(blob_url)
    if ann_response.status_code != 200:
        print(f"Impossible de récupérer les annotations depuis {blob_url}")
        return False

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

def upload_to_roboflow(api_key: str, project_name: str, presigned_url: str, img_name='', split="train"):
    """Upload an image to Roboflow."""
    API_URL = "https://api.roboflow.com"
    if img_name == '':
        img_name = presigned_url.split("/")[-1]
        img_name = img_name.split("?")[0]
    

    upload_url = "".join([
        API_URL + "/dataset/" + project_name + "/upload",
        "?api_key=" + api_key,
        "&name=" + img_name,
        "&split=" + split,
        "&image=" + urllib.parse.quote_plus(presigned_url),
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
    available_blobs = get_gcs_objects(GCS_BUCKET_NAME)
    
    print(available_blobs)
    # Optional: Filter the blobs here
    # e.g., available_blobs = [blob for blob in available_blobs if "some_condition"]
    
    response = ""

    # Upload blobs to Roboflow
    for blob in available_blobs:
        print("sending : " + blob)
        
        blob_url = get_gcs_signed_url(GCS_BUCKET_NAME, blob)
        if blob.split(".")[1].lower() == "txt":

            # image_name = os.path.splitext(blob)[0] + ".jpg"

            upload_to_roboflow_annotations_2(ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_NAME, blob, blob_url, response)


        else :
        
            response=upload_to_roboflow(ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_NAME, blob_url).json()['id']
    

