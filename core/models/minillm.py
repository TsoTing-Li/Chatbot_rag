import httpx

from tools.logger import config_logger

from .pattern import TextEmbedding

# init log
LOGGER = config_logger(
    log_name="minillm.log",
    logger_name="minillm",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class MinillmModel(TextEmbedding):
    def __init__(
        self,
        model_name: str = "all-minilm:latest",
        host: str = "localhost",
        port: int = 11434,
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
                url=self.ollama_url + "embeddings", json=data, timeout=None
            )

        if response.status_code != 200:
            LOGGER.error(f"{self.model_name} can not loaded!")
            raise RuntimeError
        LOGGER.info(f"Success init {self.model_name}!")

    def _release_model(self):
        data = {"model": self.model_name, "keep_alive": 0}
        with httpx.Client() as client:
            response = client.post(
                url=self.ollama_url + "embeddings", json=data, timeout=None
            )

        if response.status_code != 200:
            LOGGER.error(f"{self.model_name} can not released!")
        LOGGER.info(f"Success release {self.model_name}!")

    def run(self, data: str) -> list:
        # LOGGER.info(f"Input: {prompt}")
        request_data = {"model": self.model_name, "input": data}

        with httpx.Client() as client:
            response = client.post(
                url=self.ollama_url + "embed", json=request_data, timeout=None
            )

        if response.status_code == 200:
            content = response.json()
            result = content["embeddings"][0]
        return result
