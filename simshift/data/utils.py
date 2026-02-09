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
        members = zip_ref.namelist()

        # find common prefix (top-level folder)
        prefix = os.path.commonprefix(members)
        if prefix and not prefix.endswith("/"):
            prefix = prefix[: prefix.rfind("/") + 1]

        for member in members:
            target = member[len(prefix) :]  # strip prefix
            if not target:
                continue  # skip top-level directory itself

            target_path = os.path.join(local_dir, target)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zip_ref.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    # remove zip
    os.remove(zip_path)
