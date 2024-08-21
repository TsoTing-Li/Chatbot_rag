from typing import Union

from core.handler.pattern import HandlerPattern
from core.models.pattern import TextEmbedding


class TextEmb(HandlerPattern):
    """
    Handler class for generating text embeddings using a TextEmbedding model.

    This class provides the functionality to generate text embeddings using
    a provided TextEmbedding model.

    Attributes:
        model (TextEmbedding): The TextEmbedding model used for generating embeddings.

    Methods:
        run(data: Union[list, str]) -> list:
            Generate an embedding vector for the provided text data.
    """

    def __init__(self, model: TextEmbedding) -> None:
        """
        Initialize the TextEmb handler with a TextEmbedding model.

        Args:
            model (TextEmbedding): The TextEmbedding model to be used for generating embeddings.

        Raises:
            TypeError: If the provided model is not an instance of TextEmbedding or its tokenizer type is not 'text'.
        """
        super().__init__()
        self.model = self._check(model=model)

    def _check(self, model) -> TextEmbedding:
        """
        Check if the provided model is an instance of TextEmbedding and if its tokenizer type is 'text'.

        Args:
            model: The model to be checked.

        Returns:
            TextEmbedding: The model if it is a valid TextEmbedding instance with tokenizer type 'text'.

        Raises:
            TypeError: If the model is not an instance of TextEmbedding or if its tokenizer type is not 'text'.
        """
        if isinstance(model, TextEmbedding):
            # if model.type != "text":
            #     raise TypeError("The model's tokenizer is use on save pgdb!")
            return model
        raise TypeError(
            f"TextEmb must create by model which type is 'TextEmbedding'! But Input type is '{type(model)}' , More info: model name is '{model.model_name}'."
        )

    def run(self, data: Union[list, str]) -> list:
        """
        Generate an embedding vector for the provided text data.

        Args:
            data (Union[list, str]): The text data for which to generate an embedding vector.

        Returns:
            list: The generated embedding vector.
        """
        vector = self.model.run(data=data)
        return vector
