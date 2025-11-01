from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from loguru import logger


class DocumentRetriever:
    """Interface base para mecanismos de recuperação de documentos."""

    def retrieve(self, question: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Recupera uma lista de documentos relevantes."""
        raise NotImplementedError


class PostgresVectorRetriever(DocumentRetriever):
    """
    Recupera documentos vetorizados armazenados em um banco PostgreSQL com extensão PGVector.

    Args:
        collection_name (str): Nome da coleção de vetores no banco.
        connection_url (str): URL de conexão PostgreSQL.
        embedding_model (str): Modelo de embedding OpenAI (padrão: text-embedding-3-small).
    """

    def __init__(
        self,
        collection_name: str,
        connection_url: str,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        if not collection_name or not connection_url:
            raise ValueError(
                "Os parâmetros 'collection_name' e 'connection_url' devem ser fornecidos."
            )

        self.store = PGVector(
            embeddings=OpenAIEmbeddings(model=embedding_model),
            collection_name=collection_name,
            connection=connection_url,
            use_jsonb=True,
        )

    def retrieve(self, question: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Busca os k documentos mais semelhantes à pergunta informada.

        Args:
            question (str): Pergunta ou texto de consulta.
            k (int): Número de documentos a recuperar (padrão: 3).

        Returns:
            List[Tuple[Document, float]]: Lista de tuplas (documento, score).
        """
        if not question or not isinstance(question, str):
            raise ValueError("A pergunta deve ser uma string não vazia.")

        logger.info(f"Buscando informações para a pergunta: '{question}'")

        try:
            return self.store.similarity_search_with_score(question, k=k)
        except Exception:
            logger.exception("Erro ao recuperar documentos do banco vetorial")
            return []

    def retrieve_context(self, question: str, k: int = 10) -> str:
        """
        Retorna um texto consolidado com o conteúdo dos documentos mais relevantes.

        Args:
            question (str): Pergunta ou texto de consulta.
            k (int): Número de documentos a considerar (padrão: 3).

        Returns:
            str: Contexto textual concatenado dos documentos recuperados.
        """
        results = self.retrieve(question, k)
        if not results:
            logger.warning("Nenhum documento encontrado para a consulta.")
            return ""

        context_parts = [
            f"[Documento {i+1}] {doc.page_content.strip()}"
            for i, (doc, _) in enumerate(results)
        ]
        return "\n\n".join(context_parts)
