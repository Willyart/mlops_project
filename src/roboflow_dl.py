import os
from google.cloud import storage
from roboflow import Roboflow
import urllib.parse
import json
import shutil


import hashlib

def file_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Paramètres
# GCS_BUCKET_NAME = "mlopps-roboflow-ds"
# GOOGLE_APPLICATION_CREDENTIALS = "mlopps-bucket-service-account-file.json"  # Remplacez par votre chemin de fichier JSON

rf = Roboflow(api_key="4m9GkHwOJE416dHxx26p")
project = rf.workspace("alex-07bbm").project("mlops-aplvu")
versions = project.get_version_information()

version = project.version(versions[0]['id'].split("/")[-1]) #get last version of dataset available
dataset = version.download("yolov11")


def main():
    local_directory = dataset.location 

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

            dest_file_path = os.path.join(dest_directory, new_name_file)
            print(local_file_path)
            if os.path.exists(dest_file_path):
                if file_checksum(local_file_path) != file_checksum(dest_file_path):
                    shutil.move(local_file_path, dest_file_path)
                    print(f"Fichier {file} écrasé et déplacé vers {dest_file_path}")
                else:
                    print(f"Le fichier {file} existe déjà dans {dest_directory} et a le même contenu, il n'a pas été déplacé.")
            else:
                shutil.move(local_file_path, dest_file_path)
                print(f"Fichier {file} déplacé vers {dest_file_path}")

    print(f"Deleting dataset directory: {local_directory}")
    shutil.rmtree(local_directory) 

if __name__ == "__main__":
    main()
