from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "documentos"

# Modelos de OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Requisitos del enunciado: chunks de al menos 500 tokens (no caracteres) y 50 de overlap
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Fragmentos que se pasan al LLM: entre 3 y 5
TOP_K = 4
