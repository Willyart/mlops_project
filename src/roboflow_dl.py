import os
from google.cloud import storage
from roboflow import Roboflow
import urllib.parse
import json
import shutil


import hashlib

def file_checksum(file_path):
    """Calcule la somme de contrôle (MD5) d'un fichier."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Lire par morceaux de 4096 octets
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Paramètres
# GCS_BUCKET_NAME = "mlopps-roboflow-ds"
# GOOGLE_APPLICATION_CREDENTIALS = "mlopps-bucket-service-account-file.json"  # Remplacez par votre chemin de fichier JSON

# Authentification et téléchargement du dataset de Roboflow
rf = Roboflow(api_key="4m9GkHwOJE416dHxx26p")
project = rf.workspace("alex-07bbm").project("mlops-aplvu")
versions = project.get_version_information()


version = project.version(versions[0]['id'].split("/")[-1]) #get last version of dataset available
dataset = version.download("yolov11")

# # Fonction d'upload vers GCS
# def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
#     """Upload a file to Google Cloud Storage."""
#     storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
    
#     # Uploader le fichier local vers GCS
#     blob.upload_from_filename(local_file_path)
#     print(f"File {local_file_path} uploaded to {destination_blob_name}.")

# Fonction principale
def main():
    # Lister les fichiers du dataset téléchargé
    local_directory = dataset.location  # Cela donne le chemin où le dataset est téléchargé

    for root, dirs, files in os.walk(local_directory):
        print(dirs)
        for file in files:
            extension = file.split(".")[-1]

            new_name_file = file.split("JPG")[0][:-1] + "." + extension
            print(file)
            dest_directory = ""

            local_file_path = os.path.join(root, file)
            if "test" in local_file_path :
                dest_directory = "../datasets/data/test"
            elif "valid" in local_file_path :
                dest_directory = "../datasets/data/test"
            elif "train" in local_file_path :
                dest_directory = "../datasets/data/train"
            else:
                continue 

            if extension.lower() == "txt":
                dest_directory += "/labels"
            elif extension.lower() == "jpg":
                dest_directory += "/images"

            # Vérifier si le fichier existe déjà dans le répertoire de destination
            dest_file_path = os.path.join(dest_directory, new_name_file)
            print(local_file_path)
            if os.path.exists(dest_file_path):
                # Comparer les checksums pour vérifier si le contenu est identique
                if file_checksum(local_file_path) != file_checksum(dest_file_path):
                    # Si le contenu est différent, écraser le fichier
                    shutil.move(local_file_path, dest_file_path)
                    print(f"Fichier {file} écrasé et déplacé vers {dest_file_path}")
                else:
                    print(f"Le fichier {file} existe déjà dans {dest_directory} et a le même contenu, il n'a pas été déplacé.")
            else:
                # Si le fichier n'existe pas dans la destination, déplacer
                shutil.move(local_file_path, dest_file_path)
                print(f"Fichier {file} déplacé vers {dest_file_path}")


           

            destination_blob_name = file  # Utiliser le nom du fichier tel quel ou personnaliser si besoin
            # print(local_file_path)
            # Appeler la fonction d'upload pour chaque fichier
            # upload_to_gcs(local_file_path, GCS_BUCKET_NAME, destination_blob_name)



    print(f"Deleting dataset directory: {local_directory}")
    # shutil.rmtree(local_directory)  # Supprimer le dossier et son contenu

if __name__ == "__main__":
    main()
