import json
from collections.abc import Generator

import httpx

from tools.logger import config_logger

from .pattern import Text2Text

# init log
LOGGER = config_logger(
    log_name="llama.log",
    logger_name="llama",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class Llama31Model(Text2Text):
    def __init__(
        self, model_name: str = "llama3.1", host: str = "localhost", port: int = 11434
    ) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.ollama_url = f"http://{host}:{str(port)}/api/"

        self._pull_model()

    def _pull_model(self):
        data = {"name": self.model_name}

        with httpx.stream(
            "POST", url=self.ollama_url + "pull", json=data, timeout=None
        ) as response:
            if (
                response.status_code == 200
                and response.headers.get("Transfer-Encoding") == "chunked"
            ):
                LOGGER.info("Response is a streaming response")

                for chunk in response.iter_lines():
                    LOGGER.info(chunk)

    def _load_model(self):
        data = {"model": self.model_name, "keep_alive": -1}
        with httpx.Client() as client:
            response = client.post(
                url=self.ollama_url + "generate", json=data, timeout=None
            )

        if response.status_code != 200:
            LOGGER.error(f"{self.model_name} can not loaded!")
            raise RuntimeError
        LOGGER.info(f"Success init {self.model_name}!")

    def _release_model(self):
        data = {"model": self.model_name, "keep_alive": 0}
        with httpx.Client() as client:
            response = client.post(
                url=self.ollama_url + "generate", json=data, timeout=None
            )

        if response.status_code != 200:
            LOGGER.error(f"{self.model_name} can not released!")
        LOGGER.info(f"Success release {self.model_name}!")

    def chat_stream(self, request_data: dict) -> Generator[str]:
        try:
            with httpx.stream(
                "POST", url=self.ollama_url + "chat", json=request_data, timeout=None
            ) as response:
                if response.headers.get("Transfer-Encoding") == "chunked":
                    for chunk in response.iter_lines():
                        yield json.loads(chunk)["message"]["content"]
                else:
                    raise RuntimeError(json.loads(response.read().decode("utf-8")))
        except BaseException as e:
            yield f"Error occurred: {str(e)}\n\n"

    def run(self, data: list, max_tokens: int = 350) -> Generator[str]:
        LOGGER.info(
            f"Input: {[entry['content'] for entry in data if entry['role'] == 'user']}"
        )
        request_data = {
            "model": self.model_name,
            "messages": data,
            "options": {"num_predict": max_tokens},
        }
        yield from self.chat_stream(request_data=request_data)
