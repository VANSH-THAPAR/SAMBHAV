import os
import requests
import hashlib
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import time

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank

# =====================================================================================
# 1. SETUP AND CONFIGURATION
# =====================================================================================

load_dotenv()

API_KEY = "69209b0175d58128f147b0104e0b91a4f6c9ad08d9852206d28d653c3b0b48cd"
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="HackRx 6.0 Intelligent Query System",
    description="Advanced RAG API with a persistent Pinecone DB and Cohere Re-ranker.",
    version="WINNER",
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(host=os.getenv("PINECONE_INDEX_HOST"))

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./model_cache")
llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

QA_PROMPT_TEMPLATE = """
You are a highly precise Q&A bot. Your only job is to answer the user's question concisely based on the provided text.
**Context:**
---------------------
{context_str}
---------------------
**Question:** {query_str}
**Instructions:**
1. First, determine if the question can be answered with a 'Yes' or 'No'.
2. Formulate a single, direct sentence that answers the question, including critical numbers or conditions.
3. If the question is a yes/no question, begin your answer with 'Yes,' or 'No,' followed by the explanation.
4. Do not start your answer with other introductory phrases like "According to the policy...". Get straight to the point.
5. Match the style of this example: "Yes, the policy covers maternity expenses, including childbirth..."
**Answer (single sentence):**
"""
# =====================================================================================
# 2. API LOGIC
# =====================================================================================
def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the bearer token."""
    if api_key_header and api_key_header.startswith("Bearer "):
        token = api_key_header.split(" ")[1]
        if token == API_KEY:
            return token
    raise HTTPException(status_code=403, detail="Could not validate credentials")


def safe_delete(path: Path):
    """Safely deletes a file or directory without crashing the server."""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        print(f"Warning: Failed to delete {path}: {e}")

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=HackRxResponse, tags=["Submission Endpoint"])
async def run_submission_endpoint(
    request: HackRxRequest, api_key: str = Security(get_api_key)
):
    try:
        url_hash = hashlib.md5(request.documents.encode()).hexdigest()
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=url_hash)

        stats = pinecone_index.describe_index_stats()
        is_indexed = stats.namespaces.get(url_hash, {}).get('vector_count', 0) > 0

        if not is_indexed:
            print(f"Index not found in Pinecone. Creating new index for namespace: {url_hash}")
            temp_doc_path = Path(f"./temp_docs/doc_{url_hash}.pdf")
            temp_doc_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                response = requests.get(request.documents)
                response.raise_for_status()
                with open(temp_doc_path, "wb") as f:
                    f.write(response.content)

                documents = SimpleDirectoryReader(input_files=[temp_doc_path], errors='ignore').load_data()
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context, embed_model=embed_model,
                )
                print("Index upload complete. Waiting briefly for consistency...")
                time.sleep(5)
            finally:
                safe_delete(temp_doc_path)

        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        
        cohere_rerank = CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=3)
        qa_prompt_object = PromptTemplate(QA_PROMPT_TEMPLATE)

        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=15, 
            node_postprocessors=[cohere_rerank], 
            text_qa_template=qa_prompt_object,
        )
        
        answers = []
        for question in request.questions:
            print(f"Processing question: '{question}'")
            response = await query_engine.aquery(question)
            answers.append(str(response).strip())
        
        print("SUCCESS: All questions processed.")
        return HackRxResponse(answers=answers)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API is ready for HackRx 6.0!"}