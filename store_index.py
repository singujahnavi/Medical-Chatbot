from src.helper import load_pdf_files, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
 
# Load environment variables
load_dotenv()
 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
 
# Load and process documents
print("Loading PDF files...")
extracted_data = load_pdf_files(data="data/")
 
print("Splitting text into chunks...")
texts_chunk = text_split(extracted_data)
 
print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()
 
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
 
index_name = "medicalbot"
 
# Create index if it doesn't exist
existing_indexes = [i.name for i in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,        # all-MiniLM-L6-v2 produces 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")
 
# Upload documents to Pinecone
print("Uploading vectors to Pinecone... (this may take a few minutes)")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)
 
print("Done! Vectors successfully stored in Pinecone.")
