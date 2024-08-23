import os
from dataclasses import dataclass


@dataclass
class ConnectHandler:
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST")
    OLLAMA_PORT: str = os.getenv("OLLAMA_PORT")

    BART_HOST: str = os.getenv("BART_HOST")
    BART_PORT: str = os.getenv("BART_PORT")

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")

    CORE_HOST: str = os.getenv("CORE_HOST")
    CORE_PORT: str = os.getenv("CORE_PORT")


if __name__ == "__main__":
    connect_handler = ConnectHandler()
