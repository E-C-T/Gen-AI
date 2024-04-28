import os
import bs4
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
# from langchain.embeddings import VertexAIEmbeddings
from dotenv import load_dotenv
import vertexai
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


# Load environment variables
load_dotenv()

# Initialize Vertex AI
PROJECT_ID = os.getenv("PROJECT_ID")  # Read PROJECT_ID from .env
REGION = os.getenv("REGION")
BUCKET_URI = os.getenv("BUCKET_URI")

vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


# Create FastAPI application
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Conversational AI with RAG",
)

# Set CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for chat history
llm = ChatVertexAI(model_name="gemini-pro", convert_system_message_to_human=True)

# Load Webpage URLs into a vector database
urls = [
    "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
    
    ]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("p")  # Extracting all paragraphs
        )
    )

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=VertexAIEmbeddings(model_name="textembedding-gecko@latest"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Define a simple POST endpoint
add_routes(
    app,
    rag_chain,  
    path="/chat", 
)

if __name__ == "__main__":
    import uvicorn

    #uvicorn.run(app, host="localhost", port=8000) for running locally
    uvicorn.run(app, host="0.0.0.0", port=8080)