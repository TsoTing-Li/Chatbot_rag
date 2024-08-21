import os
from typing import Union

import numpy as np
from datasets import Dataset, load_from_disk

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="faiss.log",
    logger_name="faiss",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class Operator:
    """
    Faiss Operator for managing and retrieving embeddings from a Faiss index.

    This operator allows loading a dataset, saving embeddings to a Faiss index,
    loading a Faiss index, and searching for the nearest embeddings.

    Attributes:
        dataset (Dataset): The dataset containing embeddings.
        column (str): The name of the column in the dataset that contains the embeddings.

    Methods:
        save(save_path: str, em_type: str = "img") -> None:
            Save the embeddings to a Faiss index file.

        load(faiss_path: str) -> None:
            Load the Faiss index from a file.

        search(query_embedding: np.ndarray, top_k: int = 1) -> Union[int, str]:
            Search for the nearest embeddings in the Faiss index.
    """

    def __init__(
        self,
        dataset_path: str = "./database/faiss/embeddings/data",
        faiss_path: str = "./database/faiss/embeddings/img_embedding.faiss",
        column: str = "embeddings",
    ) -> None:
        """
        Initialize the Faiss Operator.

        Args:
            dataset_path (str, optional): Path to the dataset. Defaults to "./database/faiss/embeddings/data".
            column (str, optional): The name of the column in the dataset that contains the embeddings. Defaults to "embeddings".
        """
        self.column = column
        LOGGER.info("Init faiss...")
        if os.path.exists(dataset_path):
            self.dataset = load_from_disk(dataset_path)
            LOGGER.info("Success laod dataset")
        else:
            LOGGER.warn("Not load dataset")

        if os.path.exists(faiss_path):
            self.load(faiss_path=faiss_path)
            LOGGER.info("Success laod embedding data")
        else:
            LOGGER.warn("Not load embedding data")

        LOGGER.info("Success init faiss!")

    def save(
        self,
        dataset: Dataset,
        save_path: str = "./database/faiss/embeddings/img_embedding.faiss",
        em_type: str = "img",
    ) -> None:
        """
        Save the embeddings to a Faiss index file.

        Args:
            dataset (Dataset): The Dataset object.
            save_path (str): Path to save the Faiss file.
            em_type (str, optional): Type of embeddings ("img" for image embeddings, other for text embeddings). Defaults to "img".
        """
        self.dataset = dataset
        self.dataset.add_faiss_index(column=self.column)
        self.dataset.save_faiss_index(self.column, save_path)
        LOGGER.info(
            f"Success save : {save_path} , column : {self.column}, embedding type :{em_type}"
        )

    def load(self, faiss_path: str) -> None:
        """
        Load the Faiss index from a file.

        Args:
            faiss_path (str): Path to the Faiss file.
        """
        self.dataset.load_faiss_index(index_name=self.column, file=faiss_path)
        LOGGER.info(f"Success load faiss : {faiss_path} , column : {self.column}")

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> Union[int, str]:
        """
        Search for the nearest embeddings in the Faiss index.

        Args:
            query_embedding (np.ndarray): The query embedding vector.
            top_k (int, optional): The number of nearest results to return. Defaults to 1.

        Returns:
            Union[int, str]: The scores and the corresponding answers for the nearest results.
        """
        scores, answer = self.dataset.get_nearest_examples(
            self.column, query_embedding, k=top_k
        )
        LOGGER.info(
            f"Success search faiss ,vector : {query_embedding} , scores : {scores}, answer : {answer}"
        )
        return (scores, answer)


if __name__ == "__main__":
    from datasets import Dataset, load_from_disk

    from core.img_text2text.main import Service

    image_path = "./data/3ME3.png"
    clip = Service()
    img_vec = clip.get_image_vector(image_path=image_path)

    db = Operator(dataset="./embeddings/data")
    db.load(faiss_path="./embeddings/img_embedding.faiss")

    score, answer = db.search(img_vec)
    print(score, answer)
