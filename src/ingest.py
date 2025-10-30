import os
from dotenv import load_dotenv
from loguru import logger
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()
REQUIRED_ENV_VARS = ["PDF_PATH", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]


def validate_env():
    """
    Validates required environment variables.

    Checks for the presence of each environment variable specified in REQUIRED_ENV_VARS.
    Logs errors and raises EnvironmentError if any required variable is not set.

    Raises:
        EnvironmentError: If a required environment variable is missing.
    """
    logger.info("Validating required environment variables...")
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            logger.error(f"Missing environment variable: {var}")
            raise EnvironmentError(f"Missing required environment variable: {var}")
        logger.debug(f"{var} = {value}")
    logger.info("Environment variables validated successfully.")


def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF and splits its content into text chunks.

    Uses PyPDFLoader to read the PDF and RecursiveCharacterTextSplitter to split text into chunks.
    Enriches chunks with filtered metadata.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of Document objects containing split text chunks.

    Raises:
        ValueError: If no pages are found or no text chunks are created from the PDF.
    """
    logger.info(f"Loading PDF from path: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF.")

    if not docs:
        raise ValueError(f"No pages found in PDF: {pdf_path}")

    logger.info("Splitting PDF content into text chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False)
    splits = splitter.split_documents(docs)
    logger.info(f"Split PDF into {len(splits)} text chunks.")

    if not splits:
        raise ValueError("No text chunks were created from the PDF.")

    logger.info("Enriching chunks with filtered metadata...")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]
    logger.info("Document enrichment completed.")

    return enriched


def store_documents(docs):
    """
    Stores Document objects in a PGVector collection after embedding.

    Uses OpenAIEmbeddings to convert documents into embeddings and stores them using PGVector.

    Args:
        docs (list): List of Document objects to store.

    Returns:
        None

    Logs progress, number of stored documents, and success message.
    """
    logger.info("Initializing embedding and database storage pipeline...")
    model_name = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    db_url = os.getenv("DATABASE_URL")

    logger.info(f"Using OpenAI embedding model: {model_name}")
    logger.info(f"Target PGVector collection: {collection_name}")

    embeddings = OpenAIEmbeddings(model=model_name)
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=db_url,
        use_jsonb=True,
    )

    ids = [f"doc-{i}" for i in range(len(docs))]
    logger.info(f"Preparing to store {len(docs)} documents with generated IDs...")

    store.add_documents(documents=docs, ids=ids)
    logger.info(f"Successfully ingested {len(docs)} documents into PGVector collection '{collection_name}'.")


def ingest_pdf():
    """
    Orchestrates the PDF ingestion pipeline.

    Validates environment variables, loads and splits PDF content, and stores documents in PGVector.

    Returns:
        None

    Logs progress and completion messages throughout the process.
    """
    logger.info("Starting PDF ingestion pipeline...")
    validate_env()

    pdf_path = os.getenv("PDF_PATH")
    logger.info(f"Target PDF file: {pdf_path}")

    docs = load_and_split_pdf(pdf_path)
    store_documents(docs)

    logger.info("PDF ingestion pipeline completed successfully.")


if __name__ == "__main__":
    """
    Entry point for script execution.

    Runs the PDF ingestion pipeline, logs exceptions if any error occurs.
    """
    try:
        ingest_pdf()
    except Exception as e:
        logger.exception(f"PDF ingestion failed due to an error: {e}")