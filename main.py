import os
import hashlib
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import time
import asyncio
import httpx  # For async requests
import aiofiles # For async file operations

from fastapi import FastAPI, HTTPException, Security
from pydantic import BaseModel
from fastapi.security.api_key import APIKeyHeader

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRank

# =====================================================================================
# 1. SETUP AND CONFIGURATION
# =====================================================================================

load_dotenv()

# IMPORTANT: Load your API key from environment variables for security
API_KEY = os.getenv("HACKRX_API_KEY", "69209b0175d58128f147b0104e0b91a4f6c9ad08d9852206d28d653c3b0b48cd")
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="HackRx 6.0 Intelligent Query System",
    description="Optimized RAG API with a persistent Pinecone DB and Cohere Re-ranker.",
    version="WINNER_OPTIMIZED",
)

# --- LlamaIndex and Pinecone Setup (loaded once at startup) ---
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

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# This is a synchronous function that we will run in a separate thread
def sync_index_from_file(file_path: Path, vector_store: PineconeVectorStore):
    """Loads data from a file and indexes it into Pinecone synchronously."""
    print(f"Starting synchronous indexing for file: {file_path}")
    documents = SimpleDirectoryReader(input_files=[file_path], errors='ignore').load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model,
    )
    print("Synchronous indexing complete.")


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
            print(f"Index not found. Starting ingestion for namespace: {url_hash}")
            temp_doc_path = Path(f"./temp_docs/doc_{url_hash}.pdf")
            temp_doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # 1. Asynchronously download the file
                async with httpx.AsyncClient() as client:
                    response = await client.get(request.documents, follow_redirects=True, timeout=20.0)
                    response.raise_for_status()
                
                # 2. Asynchronously write the file to disk
                async with aiofiles.open(temp_doc_path, "wb") as f:
                    await f.write(response.content)

                # 3. Run the slow, CPU-bound indexing in a separate thread
                await asyncio.to_thread(sync_index_from_file, temp_doc_path, vector_store)
                
                # 4. Intelligent waiting: Poll Pinecone until the index is ready
                print("Polling Pinecone for index readiness...")
                for _ in range(15): # Poll for a max of 15 seconds
                    stats = pinecone_index.describe_index_stats()
                    count = stats.namespaces.get(url_hash, {}).get('vector_count', 0)
                    if count > 0:
                        print(f"Index is ready with {count} vectors.")
                        break
                    await asyncio.sleep(1)
                else:
                    print("Warning: Timed out waiting for index to become ready.")

            finally:
                if temp_doc_path.parent.exists():
                    shutil.rmtree(temp_doc_path.parent)

        # --- Querying Logic (runs for both new and existing documents) ---
        print(f"Proceeding to query namespace: {url_hash}")
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