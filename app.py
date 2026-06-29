from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")

print("PINECONE KEY loaded:", bool(PINECONE_API_KEY))
print("GROQ KEY loaded    :", bool(GROQ_API_KEY))

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

system_prompt = (
    "You are a medical assistant. Use the provided context to answer "
    "the user's question accurately. If you don't know, say so.\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    print(f"Bot: {answer}")
    return str(answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)