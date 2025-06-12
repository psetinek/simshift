import os
import zipfile

from huggingface_hub import hf_hub_download


def download_data(repo_id: str, filename: str, local_dir: str):
    print(f"Downloading dataset from Hugging Face: {repo_id}/{filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir
    )
    print(f"Extracting zip to: {local_dir}")

    # extract zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(local_dir)

    # remove zip
    os.remove(zip_path)
