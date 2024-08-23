import httpx

from core.models.pattern import TopicsClassification

from .pattern import HandlerPattern


class TopicsClassifier(HandlerPattern):
    """
    TopicsClassifier class.

    This class provides functionality to classify sentences into topics using a given model.

    Attributes:
        model (TopicsClassification): The model used for topic classification.
        topics (list): A list of possible topics.

    Methods:
        run(sentence: str, top_k: int = 3) -> list:
            Classify the sentence into topics and return the top-k topics.
    """

    def __init__(self, model: TopicsClassification, topics: list, url: str) -> None:
        """
        Initialize the TopicsClassifier with a model and a list of topics.

        Args:
            model (TopicsClassification): The model to be used for topic classification.
            topics (list): A list of possible topics.

        Raises:
            TypeError: If the provided model is not an instance of TopicsClassification.
        """
        super().__init__()

        self.model = self._check(model=model)
        self.topics = topics
        self.url = url

    def _check(self, model) -> TopicsClassification:
        """
        Check if the provided model is an instance of TopicsClassification.

        Args:
            model: The model to be checked.

        Returns:
            TopicsClassification: The model if it is a valid TopicsClassification instance.

        Raises:
            TypeError: If the model is not an instance of TopicsClassification.
        """
        if isinstance(model, TopicsClassification):
            return model
        raise TypeError(
            f"TopicsClassifier must create by model which type is 'TopicsClassification'! But Input type is '{type(model)}' , More info: model name is '{model.model_name}'."
        )

    def run(self, sentence: str, top_k: int = 3) -> list:
        data = {"topics": self.topics, "sentence": sentence, "top_k": top_k}
        with httpx.Client() as client:
            response = client.post(url=self.url + "/topic", json=data)
            result = response.json()
        return result["topics"]
