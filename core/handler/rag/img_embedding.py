from typing import Union

from datasets import Dataset
from PIL import Image

from core.handler.pattern import HandlerPattern
from core.models.pattern import ImageEmbedding


class ImgEmb(HandlerPattern):
    """
    Handler class for generating image embeddings using an ImageEmbedding model.

    This class provides the functionality to generate image embeddings using
    a provided ImageEmbedding model.

    Attributes:
        model (ImageEmbedding): The ImageEmbedding model used for generating embeddings.

    Methods:
        run(dataset: Dataset, column: str = "embeddings") -> Dataset:
            Generate image embeddings for a given dataset and add them to the dataset.
    """

    def __init__(self, model: ImageEmbedding) -> None:
        """
        Initialize the ImgEmb handler with an ImageEmbedding model.

        Args:
            model (ImageEmbedding): The ImageEmbedding model to be used for generating embeddings.

        Raises:
            TypeError: If the provided model is not an instance of ImageEmbedding.
        """
        super().__init__()
        self.model = self._check(model=model)

    def _check(self, model) -> ImageEmbedding:
        """
        Check if the provided model is an instance of ImageEmbedding.

        Args:
            model: The model to be checked.

        Returns:
            ImageEmbedding: The model if it is a valid ImageEmbedding instance.

        Raises:
            TypeError: If the model is not an instance of ImageEmbedding.
        """
        if isinstance(model, ImageEmbedding):
            return model
        raise TypeError(
            f"ImgEmb must create by model which type is 'ImageEmbedding'! But Input type is '{type(model)}' , More info: model name is '{model.model_name}'."
        )

    def run(self, dataset: Dataset, column: str = "embeddings") -> Dataset:
        """
        Generate image embeddings for a given dataset and add them to the dataset.

        Args:
            dataset (Dataset): A dataset object. Type must be Dataset.
            column (str, optional): The key name for the embedding vector. Defaults to "embeddings".

        Returns:
            Dataset: Dataset with added image embeddings.
        """

        def img_process_function(example):
            image = Image.open(example["images"])
            embeddings = self.model.get_image_vector(image=image)
            return {column: embeddings}

        em_dataset = dataset.map(img_process_function)
        return em_dataset
