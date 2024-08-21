from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from tools.logger import config_logger

from .pattern import TextEmbedding

# init log
LOGGER = config_logger(
    log_name="doc_embed.log",
    logger_name="doc_embed",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class DocMinillmModel(TextEmbedding):
    """
    MinillmModel class.

    This class represents a MiniLM model for text and document embedding from HuggingFace.

    Methods:
        run(data: Union[list, str]) -> list:
            Generate embeddings for the given data.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the MinillmModel with the specified model from HuggingFace.

        Args:
            model_name (str, optional): The name of the model repository on HuggingFace. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            type (Literal["text", "document"], optional): The type of embedding to use. Defaults to "text".
        """
        super().__init__(model_name)
        self.model = self._load_model(model_name=model_name)
        LOGGER.info("Success init Minillm!")

    def _load_model(self, model_name: str):
        """
        Load the MiniLM model from HuggingFace.

        Args:
            model_name (str): The repository of the model on HuggingFace.
                More information: https://docs.haystack.deepset.ai/reference/embedders-api#sentencetransformersdocumentembedder
            type (str): To choose whether to load the model for text embedding or document embedding.

        Returns:
            SentenceTransformersDocumentEmbedder: The embedding model.
        """

        embedder = SentenceTransformersDocumentEmbedder(model=model_name)

        embedder.warm_up()
        return embedder

    def run(self, data: list) -> list:
        """
        Generate embeddings for the given data.

        Args:
            data (Union[list, str]): Document or text input.

        Returns:
            list: List containing the embedding vectors.
        """
        emb_docs = self.model.run(documents=data)
        return emb_docs["documents"]
