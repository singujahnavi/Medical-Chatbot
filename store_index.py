from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")


# -------------------------------
# 1. Load PDF data
# -------------------------------

print("Loading PDFs...")

extracted_data = load_pdf_file("data/")

filter_data = filter_to_minimal_docs(extracted_data)

text_chunks = text_split(filter_data)


# -------------------------------
# 2. Create embeddings
# -------------------------------

print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()


# -------------------------------
# 3. Pinecone setup
# -------------------------------

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "medical-chatbot"


if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,
        dimension=384,   # BGE-small-en-v1.5
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created")

else:
    print("Index already exists")


index = pc.Index(index_name)


# -------------------------------
# 4. Upload vectors
# -------------------------------

stats = index.describe_index_stats()

print("Current vectors:", stats.get("total_vector_count"))


if stats.get("total_vector_count", 0) == 0:

    print("Uploading vectors...")

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print("Vectors uploaded successfully")

else:

    print("Vectors already exist. Skipping upload.")


# -------------------------------
# 5. Final stats
# -------------------------------

print(index.describe_index_stats())
print("Done!")