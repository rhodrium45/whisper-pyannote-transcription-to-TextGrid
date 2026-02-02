import requests
import os
import sys

""" Create a job on pyannote. Run this script in the format: script.py audio_file_path pyannote_api_key."""


def upload(input_path, api_key):
    # Define your media object key
    object_key = os.path.basename(input_path).split(".")[0]

    # Create the pre-signed PUT URL.
    response = requests.post(
        "https://api.pyannote.ai/v1/media/input",
        json={"url": f"media://{object_key}"},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    response.raise_for_status()
    data = response.json()
    presigned_url = data["url"]

    # Upload local file to the pre-signed URL.
    print("Uploading {0} to {1}".format(input_path, presigned_url))
    with open(input_path, "rb") as input_file:
        # Upload your local audio file.
        requests.put(presigned_url, data=input_file)

if __name__ == "__main__":
    upload(sys.argv[1], sys.argv[2])