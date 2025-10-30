import os
from retriever import DocumentRetriever, PostgresVectorRetriever
from responder import ContextualLLMResponder
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
# ----------------- RAG Chain -----------------
class QuestionAnsweringChain:
    """
    Classe que integra um retriever e um LLM para responder perguntas
    usando RAG (retrieval-augmented generation).

    Args:
        retriever (DocumentRetriever): Objeto responsável por buscar documentos relevantes.
        responder (ContextualLLMResponder): Objeto LLM que gera respostas baseadas no contexto.
    """

    def __init__(self, retriever: DocumentRetriever, responder: ContextualLLMResponder) -> None:
        self.retriever = retriever
        self.responder = responder

    def answer_question(self, question: str) -> str:
        """
        Gera uma resposta para a pergunta informada, utilizando o contexto recuperado do retriever.

        Args:
            question (str): Pergunta em linguagem natural.

        Returns:
            str: Resposta gerada pelo LLM com base no contexto.
        """
        if not question or not question.strip():
            logger.warning("Pergunta vazia recebida.")
            return "Por favor, forneça uma pergunta válida."

        context = self.retriever.retrieve_context(question)
        if not context.strip():
            logger.info("Nenhum contexto relevante encontrado para a pergunta.")
            return "Não encontrei informações suficientes para responder sua pergunta."

        return self.responder.generate_answer(context, question)


# ----------------- Helper para construir o RAG -----------------
def build_rag_chain() -> QuestionAnsweringChain:
    """
    Cria e retorna uma cadeia RAG configurada com PostgresVectorRetriever e ContextualLLMResponder.

    Returns:
        QuestionAnsweringChain: Cadeia pronta para uso no fluxo de perguntas e respostas.
    """
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    connection_url = os.getenv("DATABASE_URL")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not collection_name or not connection_url:
        raise ValueError(
            "As variáveis de ambiente PG_VECTOR_COLLECTION_NAME e DATABASE_URL devem estar definidas."
        )

    retriever = PostgresVectorRetriever(
        collection_name=collection_name,
        connection_url=connection_url,
        embedding_model=embedding_model,
    )
    responder = ContextualLLMResponder()
    return QuestionAnsweringChain(retriever, responder)


# ----------------- Função Principal Exposta -----------------
def search_prompt(question: str) -> str:
    """
    Função principal para ser usada externamente (ex: API, chatbot).
    Recebe a pergunta e retorna a resposta gerada via RAG.

    Args:
        question (str): Pergunta do usuário.

    Returns:
        str: Resposta do modelo baseada em recuperação de contexto.
    """
    try:
        rag_chain = build_rag_chain()
        return rag_chain.answer_question(question)
    except Exception:
        logger.exception("Erro ao processar a pergunta via RAG.")
        return "Não foi possível processar a pergunta no momento."
