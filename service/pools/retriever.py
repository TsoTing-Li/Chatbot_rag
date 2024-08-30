from typing import Union

from core.handler.embedding.text_embedding import TextEmb
from core.models.minillm import MinillmModel
from core.vec_db.pgvector.main import Operator as PgvecDB


class RetrieverService:
    """
    Service for retrieving data from different databases based on the input type (text or image).

    This service integrates text and image embedding models, a vector database (PgvecDB),
    a FAISS index, and a ranker for retrieving and ranking results.

    Attributes:
        text_emb_service (TextEmb): Service for generating text embeddings.
        img_emb_service (ImgEmb): Service for generating image embeddings.
        pgvec_db (PgvecDB): Vector database for text data retrieval.
        faiss (Faiss): FAISS index for image data retrieval.
        ranker (LostInTheMiddleRanker): Ranker for ranking retrieved documents.

    Methods:
        search(data: Union[str, Image.Image]) -> str:
            Search for data in the appropriate database based on the input type (text or image).
    """

    def __init__(self, text_emb_model: MinillmModel) -> None:
        """
        Initialize the RetrieverService with text and image embedding models.

        Args:
            text_emb_model (MinillmModel): The text embedding model.
            img_emb_model (ClipModel): The image embedding model.
        """
        self.text_emb_service = TextEmb(model=text_emb_model)
        self.pgvec_db = PgvecDB()
        # self.ranker = LostInTheMiddleRanker(top_k=1)

    def _search_from_pgvecdb(self, data: str) -> Union[str, None]:
        """
        Search for text data in the PgvecDB.

        Args:
            data (str): The text data to be searched.

        Returns:
            Union[str, None]: The content of the top-ranked document if found, otherwise None.
        """
        data_vector = self.text_emb_service.run(data=data)
        retriever_result = self.pgvec_db.search(query_embedding=data_vector)

        if retriever_result["documents"]:
            # rank_documents = self.ranker.run(documents=retriever_result["documents"])
            return "".join(doc.content for doc in retriever_result["documents"])
        else:
            return None

    def search(self, data: str) -> str:
        """
        Search for data in the appropriate database based on the input type (text or image).

        Args:
            data (Union[str, Image.Image]): The data to be searched, which can be either a string or an image.

        Returns:
            str: The content or description of the top-ranked result.

        Raises:
            TypeError: If the input data type is not supported.
        """

        try:
            return self._search_from_pgvecdb(data=data)
        except BaseException:
            raise TypeError("Not support type!")
