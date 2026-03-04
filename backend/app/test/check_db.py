import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

vs = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small")),
)

print("Collection count:", vs._collection.count())