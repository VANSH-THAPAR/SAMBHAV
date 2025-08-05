# 🔍 Intelligent Query–Retrieval System - HackRx 6.0

## 🌐 Hosted Endpoint

**Live Endpoint:**

```
POST https://vanshthapar-hackrx-sambhav.hf.space/hackrx/run
```

**Authorization (Required):**

```
Authorization: Bearer 69209b0175d58128f147b0104e0b91a4f6c9ad08d9852206d28d653c3b0b48cd
```

---

## 🧠 Problem Statement

Design an LLM-Powered Intelligent Query–Retrieval System that processes large documents and makes contextual decisions. The system handles real-world insurance/legal documents, parses natural language queries, and outputs structured answers with rationale.

### ✨ Key Features

* Process **PDFs**, **DOCX**, and **emails**
* Semantic search using **Pinecone**
* Clause matching with contextual reasoning
* JSON structured answers
* Token-efficient, low-latency responses
* Explainable outputs with reference clauses

---

## 🏗️ System Architecture

```
1️⃣ Input Documents (Blob URL)
        ↓
2️⃣ LLM Parser (Query Understanding)
        ↓
3️⃣ Pinecone (Embedding Search)
        ↓
4️⃣ Clause Matching (Semantic Similarity)
        ↓
5️⃣ Decision Logic (Rationale Processing)
        ↓
6️⃣ Structured JSON Response
```

---

## 📦 Tech Stack

| Component     | Tech                     |
| ------------- | ------------------------ |
| Backend API   | FastAPI                  |
| Embeddings DB | Pinecone                 |
| LLM Engine    | GPT-4 (Groq)             |
| Indexing      | LlamaIndex               |
| Asynchronous  | asyncio, httpx           |
| Deployment    | HuggingFace Spaces (CPU) |

---

## 📩 API: `/hackrx/run`

### Method: `POST`

### URL:

```
https://vanshthapar-hackrx-sambhav.hf.space/hackrx/run
```

### Headers:

```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer 69209b0175d58128f147b0104e0b91a4f6c9ad08d9852206d28d653c3b0b48cd
```

### Request Body:

```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for cataract surgery?",
    "Are organ donor medical expenses covered?"
  ]
}
```

### ✅ Sample Response:

```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "There is a waiting period of thirty-six (36) months...",
    "Yes, the policy covers maternity expenses...",
    "The policy has a specific waiting period of two (2) years...",
    "Yes, the policy indemnifies the medical expenses for the organ donor..."
  ]
}
```

---

## 🧪 Testing Instructions

* You can test the API using Postman or any frontend client.
* Ensure the `Authorization` header is present.
* Use the sample document and questions for consistent output.

---

## 🚀 Team

Built with ❤️ by Team Sambhav - HackRx 6.0

> For queries or live demo issues, contact: [vanshthapar.professional@gmail.com](mailto:vanshthapar.professional@gmail.com)

---
