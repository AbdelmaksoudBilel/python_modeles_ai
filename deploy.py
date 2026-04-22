from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])

api.upload_folder(
    repo_id="Badjou2002/python_ai_api",
    folder_path=".",
    repo_type="space"
)