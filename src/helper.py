from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from typing import List


# Extract Data From PDF File
def load_pdf_file(data):

    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    return documents



# Keep only required metadata
def filter_to_minimal_docs(
        docs: List[Document]
) -> List[Document]:

    minimal_docs = []

    for doc in docs:

        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src
                }
            )
        )

    print(f"Filtered documents: {len(minimal_docs)}")

    return minimal_docs



# Split documents into chunks
def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(
        extracted_data
    )

    print(f"Total chunks: {len(text_chunks)}")

    return text_chunks



# HuggingFace Embeddings
def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(

        # local model path
        model_name=r"C:\hf_cache\bge-small-en-v1.5",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings