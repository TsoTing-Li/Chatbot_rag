from typing import Union

from core.handler.pattern import HandlerPattern
from core.models.pattern import TextEmbedding
from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="rag_pgvector.log",
    logger_name="rag_pgvector",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class DocumentEmb(HandlerPattern):
    """
    Handler class for generating document embeddings using a TextEmbedding model.

    This class provides the functionality to generate document embeddings using
    a provided TextEmbedding model.

    Attributes:
        model (TextEmbedding): The TextEmbedding model used for generating embeddings.

    Methods:
        run(data: Union[list, str]) -> list:
            Generate an embedding vector for the provided document data.
    """

    def __init__(self, model: TextEmbedding) -> None:
        """
        Initialize the DocumentEmb handler with a TextEmbedding model.

        Args:
            model (TextEmbedding): The TextEmbedding model to be used for generating embeddings.

        Raises:
            TypeError: If the provided model is not an instance of TextEmbedding or its type is not 'document'.
        """
        super().__init__()
        self.model = self._check(model=model)
        LOGGER.info(f"Success init DocumentEmb, model:'{self.model.model_name}' ")

    def _check(self, model) -> TextEmbedding:
        """
        Check if the provided model is an instance of TextEmbedding and if its type is 'document'.

        Args:
            model: The model to be checked.

        Returns:
            TextEmbedding: The model if it is a valid TextEmbedding instance of type 'document'.

        Raises:
            TypeError: If the model is not an instance of TextEmbedding or if its type is not 'document'.
        """
        if isinstance(model, TextEmbedding):
            if model.type != "document":
                raise TypeError("The model's tokenizer is use on save pgdb!")
            return model
        LOGGER.error(
            f"DocumentEmb must create by model which type is 'TextEmbedding'! But Input type is '{type(model)}' , More info: model name is '{model.model_name}'."
        )

    def run(self, data: Union[list, str]) -> list:
        """
        Generate an embedding vector for the provided document data.

        Args:
            data (Union[list, str]): The document data for which to generate an embedding vector.

        Returns:
            list: The generated embedding vector.
        """
        vector = self.model.run(data=data)
        return vector
