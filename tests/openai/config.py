import os

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"
TEXT_EMBEDDING_CHUNK_SIZE=300
VECTOR_FIELD_NAME="content_vector"
PREFIX = "sportsdoc"  
INDEX_NAME = "f1-index"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_KEY = os.getenv("TELEGRAM_ASSISTANT")