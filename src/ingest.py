import os
import logging
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------
load_dotenv()
REQUIRED_ENV_VARS = ["PDF_PATH", "PGVECTOR_URL", "PGVECTOR_COLLECTION"]

def validate_env():
    logger.info("Validating required environment variables...")
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            logger.error(f"Missing environment variable: {var}")
            raise EnvironmentError(f"Missing required environment variable: {var}")
        logger.debug(f"{var} = {value}")
    logger.info("Environment variables validated successfully.")


# ------------------------------------------------------------
# PDF Loading & Splitting
# ------------------------------------------------------------
def load_and_split_pdf(pdf_path: str):
    logger.info(f"Loading PDF from path: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF.")

    if not docs:
        raise ValueError(f"No pages found in PDF: {pdf_path}")
    
    logger.info("Splitting PDF content into text chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,add_start_index=False)
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

# ------------------------------------------------------------
# Embedding & Storing in PGVector
# ------------------------------------------------------------
def store_documents(docs):
    logger.info("Initializing embedding and database storage pipeline...")
    model_name = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    collection_name = os.getenv("PGVECTOR_COLLECTION")
    db_url = os.getenv("PGVECTOR_URL")

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


# ------------------------------------------------------------
# Ingestion Pipeline
# ------------------------------------------------------------
def ingest_pdf():
    logger.info("Starting PDF ingestion pipeline...")
    validate_env()

    pdf_path = os.getenv("PDF_PATH")
    logger.info(f"Target PDF file: {pdf_path}")

    docs = load_and_split_pdf(pdf_path)
    store_documents(docs)

    logger.info("PDF ingestion pipeline completed successfully.")

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        logger.exception("PDF ingestion failed due to an error: %s", e)