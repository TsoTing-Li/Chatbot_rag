import httpx

from tools.logger import config_logger

from .pattern import ImageEmbedding, TextEmbedding

# init log
LOGGER = config_logger(
    log_name="clip.log",
    logger_name="clip",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class ClipModel(TextEmbedding, ImageEmbedding):
    def __init__(
        self, model_name: str = "clip", host: str = "localhost", port: int = 8889
    ) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.url = f"http://{host}:{str(port)}/"

    def _load_model(self):
        """
        Load the model, image processor, and tokenizer from HuggingFace.

        Args:
            model_name (str): The repository of the model on HuggingFace.

        Returns:
            Union[AutoModel, AutoImageProcessor, AutoTokenizer]:
                Loaded model, image processor, and tokenizer.
                More information: https://huggingface.co/docs/transformers/v4.42.0/en/autoclass_tutorial#autoimageprocessor
        """
        with httpx.Client() as client:
            response = client.get(url=self.url + "model")

        if response.status_code != 200 or response.json()["is_loaded"] is not True:
            raise RuntimeError
        assert response.json()["name"] == self.model_name
        LOGGER.info("Success init Clip!")
