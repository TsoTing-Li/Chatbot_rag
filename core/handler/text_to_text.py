from collections.abc import Generator

from core.models.pattern import Text2Text

from .pattern import HandlerPattern


class GenText(HandlerPattern):
    """
    Handler class for generating text using a Text2Text model.

    This class provides the functionality to generate text based on a given prompt
    using a Text2Text model.

    Attributes:
        model (Text2Text): The Text2Text model used for text generation.

    Methods:
        run(prompt: list, max_tokens: int = 350) -> str:
            Generate text based on the provided prompt and maximum number of tokens.
    """

    def __init__(self, model: Text2Text) -> None:
        """
        Initialize the GenText handler with a Text2Text model.

        Args:
            model (Text2Text): The Text2Text model to be used for text generation.

        Raises:
            TypeError: If the provided model is not an instance of Text2Text.
        """
        super().__init__()
        self.model = self._check(model=model)

    def _check(self, model) -> Text2Text:
        """
        Check if the provided model is an instance of Text2Text.

        Args:
            model: The model to be checked.

        Returns:
            Text2Text: The model if it is a valid Text2Text instance.

        Raises:
            TypeError: If the model is not an instance of Text2Text.
        """
        if isinstance(model, Text2Text):
            return model
        raise TypeError(
            f"GenText must create by model which type is 'Text2Text'! But Input type is '{type(model)}' , More info: model name is '{model.model_name}'."
        )

    def run(self, data: list, max_tokens: int = 350) -> Generator[str]:
        """
        Generate text based on the provided prompt and maximum number of tokens.

        Args:
            data (list): A list of prompts for text generation.
            max_tokens (int, optional): The maximum number of tokens for the generated text. Defaults to 350.

        Returns:
            Generator: The generator of text.
        """
        yield from self.model.run(data=data, max_tokens=max_tokens)
