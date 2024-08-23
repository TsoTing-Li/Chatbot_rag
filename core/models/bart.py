import httpx

from tools.logger import config_logger

from .pattern import TopicsClassification

# init log
LOGGER = config_logger(
    log_name="bart.log",
    logger_name="bart",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class BartModel(TopicsClassification):
    """
    BartModel class.

    This class represents a BART model for topic classification from HuggingFace.

    Methods:
        run(premises: List[str], hypotheses: List[str], top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
            Classify the sentences into topics and return the top-k scores and indices.
    """

    def __init__(
        self,
        model_name: str = "bart",
        host: str = "localhost",
        port: int = 8887,
    ) -> None:
        """
        Initialize the BartModel with the specified model from HuggingFace.

        Args:
            model_name (str, optional): The name of the model repository on HuggingFace. Defaults to "facebook/bart-large-mnli".
        """
        super().__init__(model_name)
        self.model_name = model_name
        self.url = f"http://{host}:{str(port)}"

    def _load_model(self) -> None:
        """
        Load the model and tokenizer from HuggingFace.

        Args:
            model_name (str): The repository of the model on HuggingFace.

        Returns:
            Tuple[AutoModelForSequenceClassification, AutoTokenizer]: The loaded model and tokenizer.
        """

        with httpx.Client() as client:
            response = client.get(url=self.url + "/model")

        if response.status_code != 200 or response.json()["is_loaded"] is not True:
            raise RuntimeError
        assert response.json()["name"] == self.model_name
        LOGGER.info("Success init Bart!")
        return
