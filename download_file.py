import os

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download


def hg_login():
    load_dotenv()
    hg_token = os.getenv("HF_API_TOKEN")
    login(hg_token)


def hg_download():
    files = {
        "facebook/bart-large-mnli": [
            "config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "sentence-transformers/all-MiniLM-L6-v2": [
            "modules.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "vocab.txt",
            "tokenizer.json",
            "special_tokens_map.json",
            "1_Pooling/config.json",
        ],
    }

    local_dir = "hf_models"
    for repo_name in files:
        model_dir = os.path.join(local_dir, repo_name)
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_name, local_dir=model_dir, allow_patterns=files[repo_name]
        )


if __name__ == "__main__":
    hg_login()
    hg_download()
