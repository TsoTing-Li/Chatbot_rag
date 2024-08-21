from abc import ABC, abstractmethod


class Model(ABC):
    """Model template, Model must come from HuggingFace.

    This abstract base class defines a template for models that must be
    derived from HuggingFace models.

    Attributes:
        model_name (str): The name of the model to load.

    Methods:
        _load_model(model_name: str):
            Load and return a model based on the model name. This method must
            be implemented by any subclass.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the Model with the given model name.

        Args:
            model_name (str): The name of the model to load.
        """
        self.model_name = model_name

    @abstractmethod
    def _load_model(self, model_name: str):
        """
        Load and return a model based on the model name.

        This abstract method must be implemented by any subclass. It should
        include the logic necessary to load the model from HuggingFace.

        Args:
            model_name (str): The name of the model to load.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Not implemented '_load_model()'!")


class Text2Text(Model):
    """Text2Text Model.

    This class represents a text-to-text model from HuggingFace, which
    typically performs tasks such as translation, summarization, and text generation.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)


class TextEmbedding(Model):
    """Text Embedding Model.

    This class represents a text embedding model from HuggingFace, which
    typically performs tasks such as generating embeddings for text for use in various downstream tasks like similarity search and clustering.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)


class ImageEmbedding(Model):
    """Image Embedding Model.

    This class represents an image embedding model from HuggingFace, which
    typically performs tasks such as generating embeddings for images for use in various downstream tasks like similarity search and clustering.

    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)


class TopicsClassification(Model):
    """Topics Classifier Model.

    This class represents a topics classifier model from HuggingFace, which
    typically performs tasks such as classifying text into predefined topics or categories.

    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
