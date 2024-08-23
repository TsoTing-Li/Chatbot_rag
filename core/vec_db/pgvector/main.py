import os
from typing import List

from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

os.environ["PG_CONN_STR"] = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
import logging

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="pgvec.log",
    logger_name="pgvec",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class Operator:
    """
    Pgvector operator for managing and retrieving documents in a vector database.

    This operator allows initializing the vector database, saving documents,
    setting up the retriever, and searching for documents based on query embeddings.

    Attributes:
        document_store (PgvectorDocumentStore): The document store for managing vector embeddings.
        document_writer (DocumentWriter): The document writer for saving documents.
        retriever (PgvectorEmbeddingRetriever): The retriever for querying the vector database.
        vector_function (str): The function to use for vector similarity.

    Methods:

        save(documents: list) -> None:
            Save the documents to the vector database.

        set_retriever(top_k: int = 2) -> None:
            Set the retriever for querying the vector database.

        search(query_embedding: List[float], filters: dict = {"operator": "AND", "conditions": [{"field": "meta.privacy", "operator": "!=", "value": "1"}]}) -> List[float]:
            Retrieve documents from the vector database based on query embeddings.
    """

    def __init__(
        self,
        recreate_table: bool = False,
        embedding_dimension: int = 384,
        vector_function: str = "cosine_similarity",
        search_strategy: str = "hnsw",
    ) -> None:
        """
        Initialize the Pgvector operator.

        Args:
            recreate_table (bool, optional): Whether to recreate the database table. Defaults to False.
            embedding_dimension (int, optional): Dimension of the embedding vectors. Defaults to 384.
            vector_function (str, optional): Function to use for vector similarity. Defaults to "cosine_similarity".
            search_strategy (str, optional): Strategy for vector search. Defaults to "hnsw".
        """

        logging.info("Init pgvector...")
        # Initializing the DocumentStore
        self.vector_function = vector_function
        self.document_store = PgvectorDocumentStore(
            embedding_dimension=embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )
        LOGGER.info(f"""Success init pgvector 
                     embedding_dimension:{embedding_dimension}
                     vector_function:{self.vector_function}
                     recreate_table:{recreate_table}
                     search_strategy:{search_strategy}""")
        self.document_writer = DocumentWriter(self.document_store)
        self.set_retriever()

    def save(self, document_vectors: list) -> None:
        """
        Save documents to the database.

        Args:
            document_vectors (list): List of documents to be saved, after document embedding.
                More information: https://docs.haystack.deepset.ai/reference/document-writers-api#documentwriter
        """
        self.document_writer.run(documents=document_vectors)

    def set_retriever(self, top_k: int = 2) -> None:
        """
        Set the retriever for querying the vector database.

        Args:
            top_k (int, optional): Maximum number of results to return. Defaults to 2.
        """
        self.retriever = PgvectorEmbeddingRetriever(
            document_store=self.document_store,
            top_k=top_k,
            vector_function=self.vector_function,
        )
        LOGGER.info(f"""Success set retriever 
                     top_k:{top_k}
                     """)

    def search(
        self,
        query_embedding: List[float],
        filters: dict = None,
    ) -> List[float]:
        """
        Retrieve documents from the vector database based on query embeddings.

        Args:
            query_embedding (List[float]): The embedding of the query to search for.
            filters (dict, optional): Filters to apply to the search.
                Defaults to { "operator": "AND", "conditions": [ {"field": "meta.privacy", "operator": "!=", "value": "1"} ] }.

        Returns:
            List[float]: List of retrieved results.
        """
        if not filters:
            filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.privacy", "operator": "!=", "value": "1"}
                ],
            }
        retriever_result = self.retriever.run(
            query_embedding=query_embedding, filters=filters
        )
        LOGGER.info(f"Retriever result : {retriever_result} ")
        return retriever_result
