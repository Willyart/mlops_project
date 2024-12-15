from inference_sdk import InferenceHTTPClient
from os import listdir
from os.path import isfile, join
import random

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    # api_url="https://detect.roboflow.com/",
    api_key="4m9GkHwOJE416dHxx26p"
)


images = [join("datasets/data/to_predict", f) for f in listdir("datasets/data/to_predict") if isfile(join("datasets/data/to_predict", f)) and f.endswith(".JPG")]

# random.shuffle(images)

# for f in images[:10]:
for f in images[:100]:
    print(f)
    response = client.run_workflow(
        workspace_name="alex-07bbm",
        # workflow_id="custom-workflow",
        workflow_id="mlops-workflow-2",
        images={"image": f},
        use_cache = False,
        parameters = {'name': "test"}
    )

    print(response[0]['roboflow_dataset_upload'])

# result = client.run_workflow(
#     workspace_name="alex-07bbm",
#     workflow_id="custom-workflow",
#     images={"image": "images/DJI_20230804090552_0020.JPG"}
# )

# print(result)