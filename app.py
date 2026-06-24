from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.prompt import system_prompt


app = Flask(__name__)

load_dotenv()


os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Load embeddings
embeddings = download_hugging_face_embeddings()


# Pinecone
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)


# OpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages(
[
    (
        "system",
        system_prompt
    ),
    (
        "human",
        """
Context:
{context}

Question:
{input}
"""
    )
]
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# LangChain 1.x RAG chain

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    |
    prompt
    |
    llm
)



@app.route("/")
def index():
    return render_template("chat.html")



@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    response = rag_chain.invoke(msg)

    return response.content



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )