import httpx
from fastapi import UploadFile

from core.handler.pattern import HandlerPattern
from core.models.pattern import ImageEmbedding

CONTENT_TYPE_MAP = {"image/jpeg": ".jpg", "image/png": ".png"}


class ImgEmb(HandlerPattern):
    """
    Handler class for generating image embeddings using an ImageEmbedding model.

    This class provides the functionality to generate image embeddings using
    a provided ImageEmbedding model.

    Attributes:
        model (ImageEmbedding): The ImageEmbedding model used for generating embeddings.

    Methods:
        run(image: PILImage) -> np.ndarray:
            Generate an embedding vector for the provided image.
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

    def run(self, data: UploadFile, host: str = "localhost", port: int = 8889):
        """
        Generate an embedding vector for the provided image.

        Args:
            image (PILImage): The image for which to generate an embedding vector.

        Returns:
            np.ndarray: The generated embedding vector.
        """
        url = f"http://{host}:{str(port)}/embed/image"
        file_extension = CONTENT_TYPE_MAP[data.content_type]
        file_name = data.filename if data.filename else f"unknown{file_extension}"
        files = {"img": (file_name, data.file, data.content_type)}
        with httpx.Client() as client:
            response = client.post(url=url, files=files)
            result = response.json()

        return result["img_vector"]
